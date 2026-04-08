"""
inference.py — Real-time deepfake detection on a video file.

Pipeline:
  Video file → frame generator (cv2, real-time, no temp files)
             → albumentations val transform
             → batched tensor
             → DeepFakeDetector
             → mean-pool probabilities (same logic as evaluate.py)
             → verdict + confidence

Usage:
    python inference.py --video path/to/video.mp4
    python inference.py --video path/to/video.mp4 --checkpoint checkpoints/best.pth
    python inference.py --video path/to/video.mp4 --frames 32 --batch-size 8 --threshold 0.5
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from config import Config
from model import DeepFakeDetector
from transforms import get_transforms


# ── Frame generator ────────────────────────────────────────────────────────────

def frame_generator(video_path: str, num_frames: int):
    """
    Yields (frame_rgb: np.ndarray, frame_index: int) for `num_frames` frames
    sampled uniformly across the video.  No temp files — purely in-memory.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = total / fps

    if total < 1:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    # Uniformly sample indices; clamp to actual frame count
    sample_count = min(num_frames, total)
    indices = np.linspace(0, total - 1, sample_count, dtype=int).tolist()

    print(f"  Video : {Path(video_path).name}")
    print(f"  Frames: {total} total  |  sampling {sample_count}  |  FPS {fps:.1f}  |  {duration:.1f}s")

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame_rgb, idx

    cap.release()


# ── Batch builder ───────────────────────────────────────────────────────────────

def batch_frames(generator, transform, batch_size: int, device: torch.device):
    """
    Consumes the frame generator, applies the albumentations transform,
    and yields (tensor_batch, list_of_frame_indices).
    """
    buf_frames, buf_indices = [], []

    for frame_rgb, idx in generator:
        tensor = transform(image=frame_rgb)["image"]   # (C, H, W) float32 tensor
        buf_frames.append(tensor)
        buf_indices.append(idx)

        if len(buf_frames) == batch_size:
            yield torch.stack(buf_frames).to(device), buf_indices
            buf_frames, buf_indices = [], []

    # Flush the last partial batch
    if buf_frames:
        yield torch.stack(buf_frames).to(device), buf_indices


# ── Core inference (mirrors evaluate.py's aggregation logic) ───────────────────

def predict_video(
    model: torch.nn.Module,
    video_path: str,
    cfg: Config,
    num_frames: int,
    batch_size: int,
    threshold: float,
    device: torch.device,
) -> dict:
    """
    Run inference on a single video.
    Returns a dict with keys: verdict, fake_prob, real_prob, frame_probs, elapsed.
    """
    transform = get_transforms("val", cfg.FACE_SIZE)   # deterministic val pipeline

    model.eval()
    all_probs = []
    t0 = time.perf_counter()

    gen = frame_generator(video_path, num_frames)

    with torch.no_grad():
        for batch, indices in batch_frames(gen, transform, batch_size, device):
            with torch.amp.autocast_mode.autocast(device.type):
                logits = model(batch)

            logits = logits.float()
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()   # P(FAKE)
            all_probs.extend(probs.tolist())

            # Live progress
            processed = len(all_probs)
            bar_len   = 30
            filled    = int(bar_len * processed / num_frames)
            bar       = "█" * filled + "░" * (bar_len - filled)
            avg_prob  = np.mean(all_probs)
            print(
                f"\r  [{bar}] {processed}/{num_frames} frames  "
                f"running P(fake)={avg_prob:.3f}",
                end="", flush=True,
            )

    print()  # newline after progress bar

    elapsed = time.perf_counter() - t0

    if not all_probs:
        raise RuntimeError("No frames could be decoded from the video.")

    # Mean-pool across all frames (same as evaluate.py's video_level_evaluate)
    fake_prob = float(np.mean(all_probs))
    real_prob = 1.0 - fake_prob
    verdict   = "FAKE" if fake_prob >= threshold else "REAL"

    return {
        "verdict"    : verdict,
        "fake_prob"  : fake_prob,
        "real_prob"  : real_prob,
        "frame_probs": all_probs,
        "elapsed"    : elapsed,
    }


# ── Pretty result printer ──────────────────────────────────────────────────────

def print_result(result: dict, video_path: str, threshold: float):
    verdict   = result["verdict"]
    fake_prob = result["fake_prob"]
    real_prob = result["real_prob"]
    elapsed   = result["elapsed"]
    n_frames  = len(result["frame_probs"])

    # Confidence bar (50 chars wide)
    bar_len  = 50
    filled   = int(bar_len * fake_prob)
    prob_bar = "█" * filled + "░" * (bar_len - filled)

    is_fake  = verdict == "FAKE"
    label    = "🔴 DEEPFAKE DETECTED" if is_fake else "🟢 REAL VIDEO"

    print()
    print("=" * 62)
    print(f"  {label}")
    print("=" * 62)
    print(f"  File      : {Path(video_path).name}")
    print(f"  Verdict   : {verdict}  (threshold={threshold:.2f})")
    print(f"  P(fake)   : {fake_prob:.4f}   [{prob_bar}]")
    print(f"  P(real)   : {real_prob:.4f}")
    print(f"  Frames    : {n_frames} sampled")
    print(f"  Time      : {elapsed:.2f}s  ({n_frames/elapsed:.1f} frames/s)")

    # Frame-level breakdown (min / mean / max)
    probs = result["frame_probs"]
    print(f"  Frame P(fake) — min:{min(probs):.3f}  mean:{np.mean(probs):.3f}  max:{max(probs):.3f}")
    print("=" * 62)
    print()


# ── Model loader ───────────────────────────────────────────────────────────────

def load_model(cfg: Config, checkpoint: str, device: torch.device) -> DeepFakeDetector:
    model = DeepFakeDetector(
        backbone_name=cfg.BACKBONE,
        pretrained=False,       # weights come from checkpoint
        dropout=cfg.DROPOUT,
        num_classes=cfg.NUM_CLASSES,
    ).to(device)

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        print(f"[WARNING] Checkpoint not found at '{ckpt_path}'. "
              "Running with random weights — results will be meaningless.")
        print("          Train the model first:  python train.py\n")
    else:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"  Checkpoint: {ckpt_path}")

    model.eval()
    return model


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Deepfake detector — predict whether a video is real or fake."
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to the input video file (mp4, avi, mov, mkv, …)"
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/best.pth",
        help="Path to the trained model checkpoint (default: checkpoints/best.pth)"
    )
    parser.add_argument(
        "--frames", type=int, default=32,
        help="Number of frames to sample from the video (default: 32)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Inference batch size (default: 8)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Probability threshold for FAKE verdict (default: 0.5)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = args.video
    if not Path(video_path).exists():
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)

    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print()
    print("── DeepFake Detector ──────────────────────────────────────")
    print(f"  Device    : {device}")

    model = load_model(cfg, args.checkpoint, device)

    print(f"  Sampling  : {args.frames} frames  |  batch {args.batch_size}")
    print("  Processing video …")
    print()

    result = predict_video(
        model      = model,
        video_path = video_path,
        cfg        = cfg,
        num_frames = args.frames,
        batch_size = args.batch_size,
        threshold  = args.threshold,
        device     = device,
    )

    print_result(result, video_path, args.threshold)

    # Exit code 1 if fake (useful for scripting)
    sys.exit(1 if result["verdict"] == "FAKE" else 0)


if __name__ == "__main__":
    main()
