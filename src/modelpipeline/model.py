"""
model.py — Improved deepfake detection architectures.

Key upgrades over the original:
  1. DualStreamDetector (PRIMARY)
     - EfficientNet-B4 spatial stream  +  SRM frequency stream
     - CBAM channel+spatial attention on spatial features
     - Multi-scale FPN feature fusion (3 backbone stages, not just the final pool)
     - Cross-attention between spatial and frequency streams
     - GeM pooling instead of global average pooling

  2. DeepFakeDetector (kept for backward-compat with existing checkpoints)
     - Original single-stream EfficientNet; unchanged API

SRM filter bank reference:
  Fridrich & Kodovsky, "Rich Models for Steganalysis of Digital Images", 2012.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np


# ── Helpers ────────────────────────────────────────────────────────────────────

class GeM(nn.Module):
    """
    Generalized Mean Pooling.  p=3 outperforms avg-pool on fine-grained tasks
    because it emphasises dominant activations over uniform spatial averages.
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):                           # (B, C, H, W)
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p).flatten(1)             # (B, C)


class ChannelAttention(nn.Module):
    """CBAM channel gate: squeeze-and-excitation with both avg and max pooling."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):                          # (B, C, H, W)
        avg  = x.mean(dim=[2, 3])
        mx   = x.amax(dim=[2, 3])
        gate = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).unsqueeze(-1).unsqueeze(-1)
        return x * gate


class SpatialAttention(nn.Module):
    """CBAM spatial gate: 7x7 conv on channel-pooled map."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):                          # (B, C, H, W)
        avg  = x.mean(dim=1, keepdim=True)
        mx   = x.amax(dim=1, keepdim=True)
        gate = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * gate


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


# ── SRM Frequency Stream ───────────────────────────────────────────────────────

class SRMConv(nn.Module):
    """
    Steganalysis Rich Model high-pass filter bank (3 fixed kernels).
    Captures pixel-level noise residuals where GAN/diffusion blending leaves
    statistical fingerprints invisible to the spatial stream.
    Weights are frozen — hand-crafted signal-processing filters.
    """
    def __init__(self):
        super().__init__()

        f1 = np.array([
            [ 0,  0,  0,  0,  0],
            [ 0, -1,  2, -1,  0],
            [ 0,  2, -4,  2,  0],
            [ 0, -1,  2, -1,  0],
            [ 0,  0,  0,  0,  0],
        ], dtype=np.float32) / 4.0

        f2 = np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8,-12,  8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1],
        ], dtype=np.float32) / 12.0

        f3 = np.array([
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  1, -2,  1,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
        ], dtype=np.float32)

        kernels = np.stack([f1, f2, f3])[:, np.newaxis]   # (3,1,5,5)
        kernels = np.repeat(kernels, 3, axis=1)            # (3,3,5,5)

        self.conv = nn.Conv2d(3, 3, kernel_size=5, padding=2, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.from_numpy(kernels))
        for p in self.conv.parameters():
            p.requires_grad = False

        self.encoder = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1, bias=False), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128),nn.ReLU(),
            nn.Conv2d(128,256,3, stride=2, padding=1, bias=False), nn.BatchNorm2d(256),nn.ReLU(),
        )
        self.pool = GeM(p=3.0)

    def forward(self, x):
        residual = self.conv(x)
        feat     = self.encoder(residual)
        return self.pool(feat)                             # (B, 256)


# ── FPN-style Multi-Scale Spatial Stream ──────────────────────────────────────

class FPNFusion(nn.Module):
    """
    Fuses features from 3 EfficientNet-B4 intermediate stages.
    EfficientNet-B4 channels: stage2=56, stage3=160, stage4=448
    """
    def __init__(self, stage_channels=(56, 160, 448), out_dim=512):
        super().__init__()
        self.cbams    = nn.ModuleList([CBAM(c) for c in stage_channels])
        self.pools    = nn.ModuleList([GeM() for _ in stage_channels])
        self.laterals = nn.ModuleList([
            nn.Sequential(nn.Linear(c, 256, bias=False), nn.LayerNorm(256))
            for c in stage_channels
        ])
        self.proj = nn.Sequential(
            nn.Linear(256 * len(stage_channels), out_dim, bias=False),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, stage_feats):
        pooled = []
        for feat, cbam, pool, lat in zip(stage_feats, self.cbams, self.pools, self.laterals):
            pooled.append(lat(pool(cbam(feat))))
        return self.proj(torch.cat(pooled, dim=1))         # (B, 512)


# ── PRIMARY MODEL: DualStreamDetector ─────────────────────────────────────────

class DualStreamDetector(nn.Module):
    """
    Dual-stream deepfake detector.

    Stream 1 — Spatial (EfficientNet-B4 multi-scale + CBAM + FPN):
      Multi-scale feature extraction with attention on manipulated regions.

    Stream 2 — Frequency (SRM filter bank):
      Noise residuals expose GAN/diffusion blending fingerprints.

    Fusion — Cross-attention:
      Spatial features query frequency evidence. Residual + LayerNorm.
      Final concat: spatial(512) + freq(256) + cross-attn(512) = 1280-d.
    """

    _EFF_B4_STAGES = (56, 160, 448)

    def __init__(self, pretrained: bool = True, dropout: float = 0.4, num_classes: int = 2):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4),
        )
        self.fpn = FPNFusion(self._EFF_B4_STAGES, out_dim=512)
        self.srm = SRMConv()

        self.freq_proj  = nn.Linear(256, 512, bias=False)
        self.cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.1, batch_first=True)
        self.attn_norm  = nn.LayerNorm(512)

        # 512 + 256 + 512 = 1280
        self.head = nn.Sequential(
            nn.Linear(1280, 512, bias=False), nn.LayerNorm(512), nn.GELU(), nn.Dropout(p=dropout),
            nn.Linear(512,  256, bias=False), nn.LayerNorm(256), nn.GELU(), nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):                          # (B, 3, H, W)
        stage_feats = self.backbone(x)
        sp_vec      = self.fpn(stage_feats)        # (B, 512)
        fr_vec      = self.srm(x)                  # (B, 256)

        q  = sp_vec.unsqueeze(1)                   # (B, 1, 512)
        kv = self.freq_proj(fr_vec).unsqueeze(1)   # (B, 1, 512)
        ca, _ = self.cross_attn(q, kv, kv)
        ca = self.attn_norm(sp_vec + ca.squeeze(1))# (B, 512)

        combined = torch.cat([sp_vec, fr_vec, ca], dim=1)  # (B, 1280)
        return self.head(combined)

    def get_feature_maps(self, x):
        return self.backbone(x)[-1]


# ── LEGACY ────────────────────────────────────────────────────────────────────

class DeepFakeDetector(nn.Module):
    """Original single-stream detector. Kept for backward-compat with old checkpoints."""
    def __init__(self, backbone_name="efficientnet_b4", pretrained=True, dropout=0.5, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        in_features   = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512), nn.SiLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def get_feature_maps(self, x):
        return self.backbone.forward_features(x)
