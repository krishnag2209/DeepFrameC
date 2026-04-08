import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


def load_manifest(frames_dir: str):
    manifest_path = Path(frames_dir) / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.csv not found at {manifest_path}. Run preprocess.py first."
        )
    return pd.read_csv(manifest_path)


def split_dataframe(df: pd.DataFrame, train=0.72, val=0.14, seed=42):
    train_df, temp_df = train_test_split(
        df, test_size=(1 - train), stratify=df["Label"], random_state=seed
    )
    val_ratio_of_temp = val / (1 - train)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_ratio_of_temp),
        stratify=temp_df["Label"], random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def make_balanced_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    label_map  = {"REAL": 0, "FAKE": 1}
    labels     = df["Label"].map(label_map).values
    class_counts = np.bincount(labels)                        # [n_real, n_fake]
    class_weights = 1.0 / class_counts                       # inverse frequency
    sample_weights = class_weights[labels]                   # per-sample weight
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


class FFppFrameDataset(Dataset):
    def __init__(self, frames_dir: str, df: pd.DataFrame, transform=None):
        self.frames_dir = Path(frames_dir)
        self.transform  = transform
        self.label_map  = {"REAL": 0, "FAKE": 1}
        self.samples    = [
            (row["Frame Path"], self.label_map[row["Label"]])
            for _, row in df.iterrows()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        img_path = self.frames_dir / rel_path

        frame = cv2.imread(str(img_path))
        if frame is None:
            # Use neighbour frame instead of a silent zero-image (wrong size, confuses model)
            alt = (idx + 1) % len(self.samples)
            frame = cv2.imread(str(self.frames_dir / self.samples[alt][0]))
            if frame is None:
                frame = np.zeros((256, 256, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.transform:
            frame = self.transform(image=frame)["image"]

        return frame, label


def build_dataloaders(cfg):
    from transforms import get_transforms

    df = load_manifest(cfg.FRAMES_DIR)
    train_df, val_df, test_df = split_dataframe(df)

    splits   = {"train": train_df, "val": val_df, "test": test_df}
    loaders  = {}

    for split, split_df in splits.items():
        ds = FFppFrameDataset(
            frames_dir=cfg.FRAMES_DIR,
            df=split_df,
            transform=get_transforms(split, cfg.FACE_SIZE),
        )

        if split == "train":
            sampler = make_balanced_sampler(split_df)
            loader  = DataLoader(
                ds,
                batch_size=cfg.BATCH_SIZE,
                sampler=sampler,              # replaces shuffle=True
                num_workers=cfg.NUM_WORKERS,
                pin_memory=True,
                persistent_workers=(cfg.NUM_WORKERS > 0),
            )
        else:
            loader = DataLoader(
                ds,
                batch_size=cfg.BATCH_SIZE,
                shuffle=False,
                num_workers=cfg.NUM_WORKERS,
                pin_memory=True,
                persistent_workers=(cfg.NUM_WORKERS > 0),
            )

        loaders[split] = loader
        print(
            f"{split}: {len(ds)} frames | "
            f"real: {(split_df['Label']=='REAL').sum()} | "
            f"fake: {(split_df['Label']=='FAKE').sum()}"
        )

    return loaders, train_df
