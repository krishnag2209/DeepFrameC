import torchaudio
import torchaudio.transforms as T
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from transforms import get_audio_transforms


class WaveFakeDataset(Dataset):
    def __init__(self, data_dir: str, df: pd.DataFrame, cfg, transform=None):
        self.data_dir = Path(data_dir)
        self.df       = df.reset_index(drop=True)
        self.cfg      = cfg
        self.transform = transform

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate = cfg.SAMPLE_RATE,
            n_fft       = cfg.N_FFT,
            hop_length  = cfg.HOP_LENGTH,
            n_mels      = cfg.N_MELS,
            power       = 2.0,
        )
        # Correct amplitude → dB conversion with top_db clamp
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=cfg.TOP_DB)

        self.label_map = {"REAL": 0, "FAKE": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        audio_path = self.data_dir / row["File Path"]
        label      = self.label_map[row["Label"]]

        waveform, sr = torchaudio.load(str(audio_path))

        # ── Mono conversion (fixes stereo → wrong channel dim bug) ────────────
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # ── Resample if needed ────────────────────────────────────────────────
        if sr != self.cfg.SAMPLE_RATE:
            waveform = T.Resample(sr, self.cfg.SAMPLE_RATE)(waveform)

        # ── Truncate / pad to fixed length ────────────────────────────────────
        n = waveform.shape[1]
        if n > self.cfg.MAX_SAMPLES:
            # Random crop during training; center crop otherwise handled by transform
            start    = torch.randint(0, n - self.cfg.MAX_SAMPLES + 1, (1,)).item()
            waveform = waveform[:, start : start + self.cfg.MAX_SAMPLES]
        elif n < self.cfg.MAX_SAMPLES:
            waveform = F.pad(waveform, (0, self.cfg.MAX_SAMPLES - n))

        # ── Mel spectrogram → dB ──────────────────────────────────────────────
        mel_spec = self.mel_spectrogram(waveform)   # (1, N_MELS, T)
        mel_spec = self.amplitude_to_db(mel_spec)   # dB, range [-top_db, 0]

        # ── Per-sample normalisation to [0, 1] ────────────────────────────────
        # Prevents backbone batchnorm from seeing wildly different value ranges
        mel_min  = mel_spec.amin()
        mel_max  = mel_spec.amax()
        mel_spec = (mel_spec - mel_min) / (mel_max - mel_min + 1e-6)

        # ── SpecAugment (train only) ───────────────────────────────────────────
        if self.transform is not None:
            mel_spec = self.transform(mel_spec)

        return mel_spec, label


def build_dataloaders(cfg):
    import os

    splits   = {}
    train_df = None

    for split in [cfg.TRAIN_SPLIT, cfg.VAL_SPLIT, cfg.TEST_SPLIT]:
        csv_path = Path(cfg.DATA_ROOT) / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected CSV at {csv_path}")
        df = pd.read_csv(csv_path)
        splits[split] = df
        if split == cfg.TRAIN_SPLIT:
            train_df = df

    transform_train = get_audio_transforms("train")
    transform_val   = get_audio_transforms("val")

    datasets = {
        cfg.TRAIN_SPLIT: WaveFakeDataset(cfg.DATA_ROOT, splits[cfg.TRAIN_SPLIT], cfg, transform=transform_train),
        cfg.VAL_SPLIT:   WaveFakeDataset(cfg.DATA_ROOT, splits[cfg.VAL_SPLIT],   cfg, transform=transform_val),
        cfg.TEST_SPLIT:  WaveFakeDataset(cfg.DATA_ROOT, splits[cfg.TEST_SPLIT],  cfg, transform=transform_val),
    }

    loaders = {
        cfg.TRAIN_SPLIT: DataLoader(
            datasets[cfg.TRAIN_SPLIT],
            batch_size  = cfg.BATCH_SIZE,
            shuffle     = True,
            num_workers = cfg.NUM_WORKERS,
            pin_memory  = True,
            drop_last   = True,
        ),
        cfg.VAL_SPLIT: DataLoader(
            datasets[cfg.VAL_SPLIT],
            batch_size  = cfg.BATCH_SIZE * 2,
            shuffle     = False,
            num_workers = cfg.NUM_WORKERS,
            pin_memory  = True,
        ),
        cfg.TEST_SPLIT: DataLoader(
            datasets[cfg.TEST_SPLIT],
            batch_size  = cfg.BATCH_SIZE * 2,
            shuffle     = False,
            num_workers = cfg.NUM_WORKERS,
            pin_memory  = True,
        ),
    }

    n_real = (train_df["Label"] == "REAL").sum()
    n_fake = (train_df["Label"] == "FAKE").sum()
    print(f"train: {len(train_df)} frames | real: {n_real} | fake: {n_fake}")

    val_df = splits[cfg.VAL_SPLIT]
    print(f"val:   {len(val_df)} frames | real: {(val_df['Label']=='REAL').sum()} | fake: {(val_df['Label']=='FAKE').sum()}")

    return loaders, train_df
