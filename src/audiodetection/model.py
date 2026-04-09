import torch
import torch.nn as nn
import timm


class AudioDeepFakeDetector(nn.Module):
    """
    EfficientNet-B4 backbone processing single-channel log-mel spectrograms.

    Key fixes vs original:
      - BatchNorm1d added before the first linear → stabilises training with
        variable-magnitude backbone outputs
      - Head depth reduced: fewer parameters = less overfitting on audio data
      - Backbone name exposed as argument for easy swapping
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b4",
        pretrained:    bool = True,
        dropout:       float = 0.3,
        num_classes:   int = 2,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained   = pretrained,
            in_chans     = 1,       # single-channel mel spectrogram
            num_classes  = 0,       # remove original head
            global_pool  = "avg",
        )
        in_features = self.backbone.num_features  # 1792 for B4

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),          # stabilise backbone output scale
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)               # (B, in_features)
        return self.classifier(features)           # (B, num_classes)
