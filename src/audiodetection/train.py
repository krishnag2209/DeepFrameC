"""
train_wav2vec2.py — Fine-tune facebook/wav2vec2-base on Fake-or-Real (FoR) dataset.

Replicates the training chain behind MelodyMachine/Deepfake-audio-detection-V2.
Expected accuracy: ~99% on FoR after 5 epochs.

Dataset layout expected (audiofolder format):
    data/FoR/
        training/
            fake/   *.wav
            real/   *.wav
        validation/
            fake/
            real/
        testing/         (optional)
            fake/
            real/

Install:
    pip install transformers datasets torch torchaudio scikit-learn accelerate

Usage:
    python train_wav2vec2.py
"""

import os
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, roc_auc_score

# ── Config ────────────────────────────────────────────────────────────────────

DATA_ROOT      = "data/FoR"          # path to downloaded FoR dataset
MODEL_ID       = "facebook/wav2vec2-base"
OUTPUT_DIR     = "checkpoints/wav2vec2-deepfake"
SAMPLE_RATE    = 16000
MAX_DURATION   = 4                   # seconds — FoR for-2sec clips are ~2s; for-norm are variable
MAX_SAMPLES    = SAMPLE_RATE * MAX_DURATION

LABEL2ID = {"fake": 0, "real": 1}
ID2LABEL = {0: "fake", 1: "real"}

# Matches V2 hyperparameters exactly
LEARNING_RATE          = 3e-5
TRAIN_BATCH_SIZE       = 32
EVAL_BATCH_SIZE        = 32
GRAD_ACCUM_STEPS       = 4           # effective batch = 128
NUM_EPOCHS             = 5
WARMUP_RATIO           = 0.1
SEED                   = 42


# ── Dataset ───────────────────────────────────────────────────────────────────

class FoRDataset(Dataset):
    """
    Scans a split directory for fake/ and real/ subfolders.
    Returns raw waveform arrays — the feature extractor handles normalisation.
    """

    AUDIO_EXTS = {".wav", ".flac", ".mp3"}

    def __init__(self, split_dir: str, feature_extractor, max_samples: int = MAX_SAMPLES):
        self.feature_extractor = feature_extractor
        self.max_samples       = max_samples
        self.samples: List[tuple] = []   # (path, label_id)

        split_path = Path(split_dir)
        for label_name, label_id in LABEL2ID.items():
            label_dir = split_path / label_name
            if not label_dir.exists():
                raise FileNotFoundError(
                    f"Expected {label_dir} — make sure your FoR layout has fake/ and real/ subdirs"
                )
            for f in sorted(label_dir.rglob("*")):
                if f.suffix.lower() in self.AUDIO_EXTS:
                    self.samples.append((str(f), label_id))

        print(f"  {split_dir}: {len(self.samples)} files | "
              f"fake={sum(1 for _,l in self.samples if l==0)} | "
              f"real={sum(1 for _,l in self.samples if l==1)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        waveform, sr = torchaudio.load(path)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

        # Truncate / pad
        n = waveform.shape[1]
        if n > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        elif n < self.max_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_samples - n))

        audio_array = waveform.squeeze().numpy()   # (T,)

        # Feature extractor: normalises + pads to uniform length
        inputs = self.feature_extractor(
            audio_array,
            sampling_rate    = SAMPLE_RATE,
            return_tensors   = "pt",
            padding          = True,
            truncation       = True,
            max_length       = self.max_samples,
        )

        return {
            "input_values": inputs["input_values"].squeeze(0),
            "labels"      : torch.tensor(label, dtype=torch.long),
        }


# ── Collator ──────────────────────────────────────────────────────────────────

@dataclass
class DataCollator:
    """Pads variable-length input_values to the longest in the batch."""
    feature_extractor: Wav2Vec2FeatureExtractor

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_values = [f["input_values"] for f in features]
        labels       = torch.stack([f["labels"] for f in features])

        batch = self.feature_extractor.pad(
            [{"input_values": v} for v in input_values],
            return_tensors = "pt",
            padding        = True,
        )
        batch["labels"] = labels
        return batch


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds  = np.argmax(logits, axis=1)
    probs  = torch.softmax(torch.tensor(logits, dtype=torch.float), dim=1)[:, 1].numpy()

    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5

    return {"accuracy": acc, "auc": auc}


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Feature extractor ─────────────────────────────────────────────────────
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        MODEL_ID,
        return_attention_mask = True,
        do_normalize          = True,
    )

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_dataset = FoRDataset(f"{DATA_ROOT}/training",   feature_extractor)
    val_dataset   = FoRDataset(f"{DATA_ROOT}/validation", feature_extractor)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels  = 2,
        label2id    = LABEL2ID,
        id2label    = ID2LABEL,
        ignore_mismatched_sizes = True,
    )

    # Freeze feature encoder (CNN layers) — only fine-tune transformer layers
    # This is standard practice for wav2vec2 fine-tuning on small datasets
    model.freeze_feature_encoder()

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable:    {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Training args — matches V2 hyperparameters exactly ────────────────────
    training_args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = TRAIN_BATCH_SIZE,
        per_device_eval_batch_size  = EVAL_BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM_STEPS,
        learning_rate               = LEARNING_RATE,
        warmup_ratio                = WARMUP_RATIO,
        lr_scheduler_type           = "cosine",
        optim                       = "adamw_torch",
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "accuracy",
        greater_is_better           = True,
        logging_steps               = 50,
        seed                        = SEED,
        fp16                        = torch.cuda.is_available(),
        dataloader_num_workers      = 4,
        report_to                   = "none",           # set to "tensorboard" if you want logs
    )

    data_collator = DataCollator(feature_extractor=feature_extractor)

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        data_collator   = data_collator,
        compute_metrics = compute_metrics,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting training...")
    trainer.train()

    # ── Save final model + feature extractor ─────────────────────────────────
    trainer.save_model(OUTPUT_DIR)
    feature_extractor.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")

    # ── Final eval ────────────────────────────────────────────────────────────
    results = trainer.evaluate()
    print(f"\nFinal Val Accuracy : {results['eval_accuracy']:.4f}")
    print(f"Final Val AUC      : {results['eval_auc']:.4f}")


if __name__ == "__main__":
    train()
