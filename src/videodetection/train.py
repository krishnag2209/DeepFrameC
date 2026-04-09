"""
train_deepfake_video.py — Fine-tune EfficientNet-B4 on FaceForensics++ C23.

Uses Config from config.py exactly as provided.

FF++ layout expected:
    <DATA_ROOT>/
        original_sequences/youtube/<COMPRESSION>/videos/     ← REAL .mp4
        manipulated_sequences/<method>/<COMPRESSION>/videos/ ← FAKE .mp4

Run order:
    # Step 1 — extract + crop faces (once, ~30-60 min)
    python train_deepfake_video.py --extract

    # Step 2 — train
    python train_deepfake_video.py

Install:
    pip install torch torchvision timm opencv-python-headless facenet-pytorch scikit-learn tqdm pandas
"""

import os
import cv2
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score
import timm

from config import Config


# ── Face extraction ───────────────────────────────────────────────────────────

def extract_faces(cfg: Config):
    """
    Samples cfg.FRAMES_PER_VIDEO frames from each video, detects faces with MTCNN,
    saves crops as JPEGs under cfg.FRAMES_DIR/REAL/<video_id>/ and FAKE/<video_id>/.
    Falls back to centre-crop when MTCNN finds no face.
    Writes cfg.FRAMES_DIR/manifest.csv on completion.
    """
    from facenet_pytorch import MTCNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn  = MTCNN(
        image_size   = cfg.FACE_SIZE,
        margin       = 40,
        device       = device,
        keep_all     = False,
        post_process = False,   # returns uint8 tensor
    )

    data_root  = Path(cfg.DATA_ROOT)
    frames_dir = Path(cfg.FRAMES_DIR)

    # ── Collect video paths ───────────────────────────────────────────────────
    video_entries = []   # (Path, label)

    real_dir = data_root / "original_sequences" / "youtube" / cfg.COMPRESSION / "videos"
    if real_dir.exists():
        for v in sorted(real_dir.glob("*.mp4")):
            video_entries.append((v, "REAL"))
    else:
        print(f"WARNING: real videos not found at {real_dir}")

    for method in cfg.MANIPULATION_TYPES:
        fake_dir = data_root / "manipulated_sequences" / method / cfg.COMPRESSION / "videos"
        if fake_dir.exists():
            for v in sorted(fake_dir.glob("*.mp4")):
                video_entries.append((v, "FAKE"))
        else:
            print(f"WARNING: {method} not found at {fake_dir}")

    print(f"\nVideos found : {len(video_entries)}")
    print(f"  REAL : {sum(1 for _,l in video_entries if l=='REAL')}")
    print(f"  FAKE : {sum(1 for _,l in video_entries if l=='FAKE')}")

    records = []

    for video_path, label in tqdm(video_entries, desc="Extracting faces", unit="video"):
        cap   = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            continue

        indices  = np.linspace(0, total - 1, cfg.FRAMES_PER_VIDEO, dtype=int)
        save_dir = frames_dir / label / video_path.stem
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, frame_idx in enumerate(indices):
            save_path = save_dir / f"{i:03d}.jpg"

            if save_path.exists():
                records.append({
                    "path": str(save_path), "label": label,
                    "video_id": video_path.stem,
                })
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                face_tensor = mtcnn(frame_rgb)   # uint8 (3, H, W) or None
            except Exception:
                face_tensor = None

            if face_tensor is not None:
                face_np = face_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                cv2.imwrite(str(save_path), cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR))
            else:
                # Centre-crop fallback
                h, w    = frame_rgb.shape[:2]
                s       = min(h, w)
                y0, x0  = (h - s) // 2, (w - s) // 2
                cropped = cv2.resize(frame_rgb[y0:y0+s, x0:x0+s],
                                     (cfg.FACE_SIZE, cfg.FACE_SIZE))
                cv2.imwrite(str(save_path), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

            records.append({
                "path": str(save_path), "label": label,
                "video_id": video_path.stem,
            })

        cap.release()

    manifest = pd.DataFrame(records)
    manifest.to_csv(frames_dir / "manifest.csv", index=False)
    print(f"\nExtracted {len(manifest)} face crops → {frames_dir}/manifest.csv")
    return manifest


# ── Train / val / test split ──────────────────────────────────────────────────

def make_splits(cfg: Config, seed: int = 42) -> pd.DataFrame:
    """
    Splits by video_id to prevent frame-level leakage.
    Uses FF++ official JSON splits if present under DATA_ROOT/splits/,
    otherwise falls back to 72/10/18 random split.
    """
    manifest_path = Path(cfg.FRAMES_DIR) / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.csv not found at {manifest_path}\n"
            f"Run with --extract first."
        )

    df         = pd.read_csv(manifest_path)
    split_json = Path(cfg.DATA_ROOT) / "splits" / "train.json"

    if split_json.exists():
        df = _apply_official_splits(df, cfg)
    else:
        df = _random_split(df, seed)

    for s in [cfg.TRAIN_SPLIT, cfg.VAL_SPLIT, cfg.TEST_SPLIT]:
        sub    = df[df["split"] == s]
        n_real = (sub["label"] == "REAL").sum()
        n_fake = (sub["label"] == "FAKE").sum()
        print(f"{s:5s}: {len(sub):6d} frames | REAL {n_real:5d} | FAKE {n_fake:5d}")

    return df


def _random_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    import random
    random.seed(seed)
    video_ids = sorted(df["video_id"].unique())
    random.shuffle(video_ids)
    n        = len(video_ids)
    n_val    = max(1, int(n * 0.10))
    n_test   = max(1, int(n * 0.18))
    val_ids  = set(video_ids[:n_val])
    test_ids = set(video_ids[n_val : n_val + n_test])
    df       = df.copy()
    df["split"] = df["video_id"].apply(
        lambda v: "val" if v in val_ids else ("test" if v in test_ids else "train")
    )
    return df


def _apply_official_splits(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Map FF++ official train/val/test JSON files to manifest rows."""
    split_dir  = Path(cfg.DATA_ROOT) / "splits"
    id_to_split = {}
    for split_name in [cfg.TRAIN_SPLIT, cfg.VAL_SPLIT, cfg.TEST_SPLIT]:
        json_path = split_dir / f"{split_name}.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            pairs = json.load(f)
        for pair in pairs:
            for vid_id in pair:
                id_to_split[vid_id] = split_name

    def resolve(video_id):
        base = video_id.split("_")[0]   # fake IDs look like "000_003"
        return id_to_split.get(base, "train")

    df         = df.copy()
    df["split"] = df["video_id"].apply(resolve)
    return df


# ── Dataset ───────────────────────────────────────────────────────────────────

def get_transforms(split: str, face_size: int) -> transforms.Compose:
    # ImageNet stats — correct for timm models pretrained on ImageNet
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.Resize((face_size, face_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((face_size, face_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class FaceForensicsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str, cfg: Config):
        self.df        = df[df["split"] == split].reset_index(drop=True)
        self.transform = get_transforms(split, cfg.FACE_SIZE)
        self.label_map = {"REAL": 0, "FAKE": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["path"])
        if img is None:
            img = np.zeros((self.transform.transforms[1].size[0],
                            self.transform.transforms[1].size[0], 3), dtype=np.uint8)
        img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img   = self.transform(img)
        label = self.label_map[row["label"]]
        return img, label


# ── Model ─────────────────────────────────────────────────────────────────────

class DeepfakeDetector(nn.Module):
    """
    cfg.BACKBONE (default efficientnet_b4) pretrained on ImageNet,
    with a stabilised classification head.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.BACKBONE,
            pretrained  = cfg.PRETRAINED,
            num_classes = 0,
            global_pool = "avg",
        )
        in_features = self.backbone.num_features   # 1792 for efficientnet_b4

        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=cfg.DROPOUT),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=cfg.DROPOUT / 2),
            nn.Linear(512, cfg.NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, desc="Val"):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=desc, leave=False, unit="batch"):
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast(device.type):
                logits = model(imgs)
            probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds      = (all_probs >= 0.5).astype(int)
    acc        = accuracy_score(all_labels, preds)
    auc        = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    return acc, auc


# ── Train ─────────────────────────────────────────────────────────────────────

def train(cfg: Config):
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Backbone: {cfg.BACKBONE}  |  Input: {cfg.FACE_SIZE}px  |  Epochs: {cfg.EPOCHS}")

    df = make_splits(cfg)

    train_ds = FaceForensicsDataset(df, cfg.TRAIN_SPLIT, cfg)
    val_ds   = FaceForensicsDataset(df, cfg.VAL_SPLIT,   cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg.BATCH_SIZE,
        shuffle     = True,
        num_workers = cfg.NUM_WORKERS,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg.BATCH_SIZE * 2,
        shuffle     = False,
        num_workers = cfg.NUM_WORKERS,
        pin_memory  = True,
    )

    model = DeepfakeDetector(cfg).to(device)
    print(f"Params  : {sum(p.numel() for p in model.parameters()):,}")

    # Class weights — 4 fake methods gives ~4:1 fake:real ratio
    train_sub = df[df["split"] == cfg.TRAIN_SPLIT]
    n_real    = (train_sub["label"] == "REAL").sum()
    n_fake    = (train_sub["label"] == "FAKE").sum()
    total     = n_real + n_fake
    weights   = torch.tensor(
        [total / max(n_real, 1), total / max(n_fake, 1)], dtype=torch.float
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.LABEL_SMOOTHING)

    # Backbone at cfg.LR/10, head at cfg.LR
    optimizer = AdamW([
        {"params": model.backbone.parameters(), "lr": cfg.LR / 10},
        {"params": model.head.parameters(),     "lr": cfg.LR},
    ], weight_decay=cfg.WEIGHT_DECAY)

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=cfg.LR / 100)
    scaler    = torch.amp.GradScaler(device.type)

    best_auc = 0.0

    for epoch in range(cfg.EPOCHS):
        model.train()
        train_loss = 0.0
        correct    = 0
        n_total    = 0

        bar = tqdm(
            train_loader,
            desc  = f"Epoch {epoch+1:02d}/{cfg.EPOCHS} [Train]",
            unit  = "batch",
            leave = True,
        )

        for imgs, labels in bar:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device.type):
                logits = model(imgs)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            n_total    += labels.size(0)

            bar.set_postfix(
                loss = f"{loss.item():.4f}",
                acc  = f"{correct / n_total:.4f}",
            )

        scheduler.step()

        val_acc, val_auc = evaluate(
            model, val_loader, device,
            desc = f"Epoch {epoch+1:02d}/{cfg.EPOCHS} [Val]",
        )

        print(
            f"Epoch {epoch+1:02d}/{cfg.EPOCHS} | "
            f"Loss {train_loss / len(train_loader):.4f} | "
            f"Train Acc {correct / n_total:.4f} | "
            f"Val Acc {val_acc:.4f} | Val AUC {val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, "best.pth"))
            print(f"  Saved best model (AUC: {best_auc:.4f})")

        torch.save({
            "epoch"    : epoch,
            "model"    : model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_auc" : best_auc,
        }, os.path.join(cfg.CHECKPOINT_DIR, "last.pth"))

    print(f"\nTraining complete. Best Val AUC: {best_auc:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extract", action="store_true",
        help="Extract + crop faces from videos before training",
    )
    args = parser.parse_args()
    cfg  = Config()

    if args.extract:
        extract_faces(cfg)

    train(cfg)
