"""
Deepfake Video Detection - Inference Script
Model: prithivMLmods/Deep-Fake-Detector-v2-Model (ViT-base, ~92% accuracy)
Labels: "Realism" (real) | "Deepfake" (fake)

Usage:
    python inference.py --video path/to/video.mp4
    python inference.py --video path/to/video.mp4 --frames 30 --device cuda
    python inference.py --video path/to/video.mp4 --frames 50 --batch_size 16

Requirements:
    pip install torch torchvision transformers opencv-python Pillow tqdm
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor


# ── Constants ────────────────────────────────────────────────────────────────

MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
DEFAULT_FRAMES = 32       # number of evenly-sampled frames to analyse
DEFAULT_BATCH  = 8        # frames per forward pass
DEEPFAKE_LABEL = "Deepfake"
REAL_LABEL     = "Realism"


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(device: torch.device):
    print(f"[*] Loading model  : {MODEL_ID}")
    processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    model     = ViTForImageClassification.from_pretrained(MODEL_ID)
    model.to(device).eval()
    print(f"[*] Device         : {device}")
    return processor, model


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(video_path: str, n_frames: int) -> list[Image.Image]:
    """
    Uniformly sample `n_frames` frames from the video.
    Returns a list of PIL RGB images.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[!] Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    dur   = total / fps if fps > 0 else 0.0

    print(f"[*] Video          : {video_path}")
    print(f"[*] Total frames   : {total}  |  FPS: {fps:.2f}  |  Duration: {dur:.2f}s")

    if total == 0:
        sys.exit("[!] Video has 0 frames.")

    indices = np.linspace(0, total - 1, num=min(n_frames, total), dtype=int)
    frames  = []

    for idx in tqdm(indices, desc="Extracting frames", unit="frame"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, bgr = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))

    cap.release()
    print(f"[*] Extracted      : {len(frames)} frames")
    return frames


# ── Per-frame inference ───────────────────────────────────────────────────────

def run_inference(
    frames: list[Image.Image],
    processor: ViTImageProcessor,
    model: ViTForImageClassification,
    device: torch.device,
    batch_size: int,
) -> list[dict]:
    """
    Run batched inference.
    Returns a list of per-frame dicts:
        {"label": str, "deepfake_prob": float, "real_prob": float}
    """
    id2label = model.config.id2label   # {0: "Deepfake", 1: "Realism"} (or vice-versa)

    # Find the index corresponding to each label
    deepfake_idx = next(k for k, v in id2label.items() if v == DEEPFAKE_LABEL)
    real_idx     = next(k for k, v in id2label.items() if v == REAL_LABEL)

    results = []

    for i in tqdm(range(0, len(frames), batch_size), desc="Running inference", unit="batch"):
        batch_imgs = frames[i : i + batch_size]
        inputs = processor(images=batch_imgs, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(**inputs).logits          # (B, num_classes)
            probs  = F.softmax(logits, dim=-1)       # (B, num_classes)

        for j in range(probs.shape[0]):
            p = probs[j].cpu().tolist()
            deepfake_p = p[deepfake_idx]
            real_p     = p[real_idx]
            label      = DEEPFAKE_LABEL if deepfake_p > real_p else REAL_LABEL
            results.append({
                "label":         label,
                "deepfake_prob": deepfake_p,
                "real_prob":     real_p,
            })

    return results


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(per_frame: list[dict]) -> dict:
    """
    Aggregate per-frame results into a single video-level verdict.

    Strategy:
      - Mean pooling of deepfake probabilities across all frames.
      - Verdict: "Deepfake" if mean deepfake_prob > 0.5, else "Realism".
      - Confidence = max(mean_deepfake_prob, mean_real_prob) * 100.
    """
    deepfake_probs = [r["deepfake_prob"] for r in per_frame]
    real_probs     = [r["real_prob"]     for r in per_frame]

    mean_deepfake = float(np.mean(deepfake_probs))
    mean_real     = float(np.mean(real_probs))

    verdict    = DEEPFAKE_LABEL if mean_deepfake > mean_real else REAL_LABEL
    confidence = max(mean_deepfake, mean_real) * 100.0

    n_deepfake_frames = sum(1 for r in per_frame if r["label"] == DEEPFAKE_LABEL)
    frame_ratio       = n_deepfake_frames / len(per_frame) * 100.0

    return {
        "verdict":            verdict,
        "confidence":         confidence,
        "mean_deepfake_prob": mean_deepfake * 100.0,
        "mean_real_prob":     mean_real     * 100.0,
        "deepfake_frames":    n_deepfake_frames,
        "total_frames":       len(per_frame),
        "deepfake_frame_pct": frame_ratio,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(agg: dict, per_frame: list[dict], verbose: bool):
    bar   = "═" * 52
    label = agg["verdict"]
    conf  = agg["confidence"]

    print(f"\n{bar}")
    print(f"  VERDICT    : {label.upper()}")
    print(f"  CONFIDENCE : {conf:.2f}%")
    print(f"{bar}")
    print(f"  Mean deepfake probability : {agg['mean_deepfake_prob']:.2f}%")
    print(f"  Mean real    probability  : {agg['mean_real_prob']:.2f}%")
    print(f"  Deepfake frames           : {agg['deepfake_frames']} / {agg['total_frames']}"
          f"  ({agg['deepfake_frame_pct']:.1f}%)")
    print(f"{bar}\n")

    if verbose:
        print("Per-frame breakdown:")
        print(f"  {'Frame':>6}  {'Label':<12}  {'Deepfake%':>10}  {'Real%':>8}")
        print("  " + "-" * 44)
        for i, r in enumerate(per_frame):
            print(f"  {i+1:>6}  {r['label']:<12}  "
                  f"{r['deepfake_prob']*100:>9.2f}%  {r['real_prob']*100:>7.2f}%")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Deepfake video detection using prithivMLmods/Deep-Fake-Detector-v2-Model"
    )
    p.add_argument("--video",      required=True,             help="Path to input video file")
    p.add_argument("--frames",     type=int, default=DEFAULT_FRAMES,
                   help=f"Number of frames to sample (default: {DEFAULT_FRAMES})")
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH,
                   help=f"Inference batch size (default: {DEFAULT_BATCH})")
    p.add_argument("--device",     default="auto",
                   choices=["auto", "cpu", "cuda", "mps"],
                   help="Compute device (default: auto)")
    p.add_argument("--verbose",    action="store_true",
                   help="Print per-frame probabilities")
    return p.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = resolve_device(args.device)

    if not Path(args.video).is_file():
        sys.exit(f"[!] File not found: {args.video}")

    processor, model = load_model(device)
    frames           = extract_frames(args.video, args.frames)
    per_frame        = run_inference(frames, processor, model, device, args.batch_size)
    agg              = aggregate(per_frame)
    print_report(agg, per_frame, args.verbose)

    # Exit code: 1 if deepfake detected, 0 if real
    sys.exit(1 if agg["verdict"] == DEEPFAKE_LABEL else 0)


if __name__ == "__main__":
    main()
