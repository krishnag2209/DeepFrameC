"""
infer.py — Run MelodyMachine/Deepfake-audio-detection-V2 on your WaveFake data.

Install deps first:
    pip install transformers torchaudio torch scikit-learn pandas tqdm

Usage:
    # Single file
    python infer.py --file path/to/audio.wav

    # Entire test split (produces predictions CSV + prints AUC/accuracy)
    python infer.py --csv data/WaveFake/test.csv --data_root data/WaveFake
"""

import argparse
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline, AutoFeatureExtractor, AutoModelForAudioClassification
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID   = "MelodyMachine/Deepfake-audio-detection-V2"
SAMPLE_RATE = 16000          # wav2vec2 expects 16 kHz
MAX_SECONDS = 10             # clip longer files to avoid OOM; model handles up to ~30s
MAX_SAMPLES = SAMPLE_RATE * MAX_SECONDS


# ── Audio loader ──────────────────────────────────────────────────────────────

def load_audio(path: str) -> np.ndarray:
    """Load any audio file → mono float32 numpy array at 16 kHz."""
    waveform, sr = torchaudio.load(str(path))

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

    # Clip to MAX_SAMPLES
    waveform = waveform[:, :MAX_SAMPLES]

    return waveform.squeeze().numpy()   # (T,) float32


# ── Load model once ───────────────────────────────────────────────────────────

def load_pipeline(device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {MODEL_ID} on {device} ...")
    clf = pipeline(
        "audio-classification",
        model     = MODEL_ID,
        device    = 0 if device == "cuda" else -1,
    )
    print(f"Labels: {clf.model.config.id2label}")
    return clf


# ── Single-file inference ─────────────────────────────────────────────────────

def predict_file(clf, audio_path: str) -> dict:
    audio  = load_audio(audio_path)
    # pipeline accepts raw numpy array + sampling_rate
    result = clf({"array": audio, "sampling_rate": SAMPLE_RATE}, top_k=2)
    # result = [{"label": "FAKE", "score": 0.99}, {"label": "REAL", "score": 0.01}]
    scores = {r["label"].upper(): r["score"] for r in result}
    label  = max(scores, key=scores.get)
    return {"label": label, "fake_score": scores.get("FAKE", 0.0), "real_score": scores.get("REAL", 0.0)}


# ── Batch evaluation on CSV ───────────────────────────────────────────────────

def evaluate_csv(clf, csv_path: str, data_root: str, output_csv: str = "predictions.csv"):
    df        = pd.read_csv(csv_path)
    data_root = Path(data_root)

    label_map = {"REAL": 0, "FAKE": 1}

    pred_labels = []
    fake_scores = []
    true_labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating", unit="file"):
        path  = data_root / row["File Path"]
        truth = row["Label"].upper()

        try:
            pred = predict_file(clf, str(path))
            pred_labels.append(pred["label"])
            fake_scores.append(pred["fake_score"])
            true_labels.append(truth)
        except Exception as e:
            print(f"  SKIP {path}: {e}")
            continue

    # ── Metrics ───────────────────────────────────────────────────────────────
    y_true  = [label_map[l] for l in true_labels]
    y_pred  = [label_map[l] for l in pred_labels]
    y_score = fake_scores

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan")

    print(f"\nAccuracy : {acc:.4f}")
    print(f"AUC      : {auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["REAL", "FAKE"]))

    # ── Save predictions ──────────────────────────────────────────────────────
    out_df = df.iloc[:len(true_labels)].copy()
    out_df["Predicted"] = pred_labels
    out_df["FakeScore"] = fake_scores
    out_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",      type=str, help="Single audio file to classify")
    parser.add_argument("--csv",       type=str, help="CSV file with File Path + Label columns")
    parser.add_argument("--data_root", type=str, default="data/WaveFake", help="Root dir for CSV paths")
    parser.add_argument("--output",    type=str, default="predictions.csv")
    parser.add_argument("--device",    type=str, default=None, help="cuda or cpu")
    args = parser.parse_args()

    clf = load_pipeline(args.device)

    if args.file:
        result = predict_file(clf, args.file)
        print(f"\nFile   : {args.file}")
        print(f"Label  : {result['label']}")
        print(f"FAKE   : {result['fake_score']:.4f}")
        print(f"REAL   : {result['real_score']:.4f}")

    elif args.csv:
        evaluate_csv(clf, args.csv, args.data_root, args.output)

    else:
        parser.print_help()
