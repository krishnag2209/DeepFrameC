"""
inference.py — Deepfake video inference using fine-tuned Swin-B checkpoint.

Modes:
    Single video  : python inference.py --video path/to/video.mp4
    Directory     : python inference.py --video_dir path/to/videos/
    Single image  : python inference.py --image path/to/face.jpg
    CSV batch     : python inference.py --manifest path/to/manifest.csv

Output:
    Per-video verdict (REAL / FAKE), fake probability, per-frame breakdown.
    Optional --output results.json to persist results.

Strategy:
    - Extract cfg.FRAMES_PER_VIDEO evenly-spaced frames per video
    - Detect + crop faces with MTCNN (centre-crop fallback)
    - Run Swin-B forward pass on each frame
    - Aggregate fake probabilities across frames via mean pooling
    - Threshold at 0.5 for final verdict

Install:
    pip install torch torchvision transformers opencv-python-headless \
                facenet-pytorch tqdm pandas
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
from torchvision import transforms
from transformers import SwinForImageClassification

from config import Config


# ── Constants ─────────────────────────────────────────────────────────────────

HF_MODEL_ID    = "microsoft/swin-base-patch4-window7-224"
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]
LABEL_MAP      = {0: "REAL", 1: "FAKE"}


# ── Model ─────────────────────────────────────────────────────────────────────

class SwinDeepfakeDetector(nn.Module):
    """
    Must match the architecture defined in train.py exactly so that
    state_dict keys align with the saved checkpoint.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.backbone = SwinForImageClassification.from_pretrained(
            HF_MODEL_ID,
            num_labels              = cfg.NUM_CLASSES,
            ignore_mismatched_sizes = True,
        )
        in_features = self.backbone.swin.num_features   # 1024 for Swin-B

        self.backbone.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=cfg.DROPOUT),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=cfg.DROPOUT / 2),
            nn.Linear(512, cfg.NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(pixel_values=x).logits


def load_model(checkpoint_path: str, cfg: Config, device: torch.device) -> nn.Module:
    model = SwinDeepfakeDetector(cfg).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Handle both raw state_dict saves and wrapped checkpoint dicts
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)

    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


# ── Preprocessing ─────────────────────────────────────────────────────────────

def get_inference_transform(face_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((face_size, face_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_mtcnn(face_size: int, device: torch.device):
    from facenet_pytorch import MTCNN
    return MTCNN(
        image_size   = face_size,
        margin       = 40,
        device       = device,
        keep_all     = False,
        post_process = False,
    )


def crop_face(frame_rgb: np.ndarray, mtcnn, face_size: int) -> np.ndarray:
    """
    Returns a (face_size, face_size, 3) uint8 RGB crop.
    Attempts MTCNN detection; falls back to centre-crop on failure.
    """
    try:
        face_tensor = mtcnn(frame_rgb)
    except Exception:
        face_tensor = None

    if face_tensor is not None:
        return face_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    # Centre-crop fallback
    h, w   = frame_rgb.shape[:2]
    s      = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return cv2.resize(frame_rgb[y0:y0+s, x0:x0+s], (face_size, face_size))


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(video_path: str, n_frames: int, mtcnn, face_size: int) -> list:
    """
    Returns list of (frame_index, np.ndarray[H,W,3] RGB) tuples.
    Silently skips unreadable frames.
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    crops   = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crop      = crop_face(frame_rgb, mtcnn, face_size)
        crops.append((int(idx), crop))

    cap.release()
    return crops


# ── Inference core ────────────────────────────────────────────────────────────

@torch.inference_mode()
def predict_frames(
    frames: list,
    model: nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int = 8,
) -> list:
    """
    Args:
        frames : list of (frame_index, np.ndarray RGB) from extract_frames
    Returns:
        list of {"frame_idx": int, "fake_prob": float, "pred": str}
    """
    results = []

    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i : i + batch_size]
        tensors      = torch.stack([transform(f) for _, f in batch_frames]).to(device)

        with torch.amp.autocast(device.type):
            logits = model(tensors)

        probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()

        for (frame_idx, _), prob in zip(batch_frames, probs):
            results.append({
                "frame_idx" : frame_idx,
                "fake_prob" : round(float(prob), 4),
                "pred"      : LABEL_MAP[int(prob >= 0.5)],
            })

    return results


def aggregate_verdict(frame_results: list, threshold: float = 0.5) -> dict:
    """
    Mean-pool fake probabilities across frames → final verdict.
    Also returns frame-level majority vote for cross-checking.
    """
    if not frame_results:
        return {"verdict": "UNKNOWN", "mean_fake_prob": None, "frame_fake_votes": 0,
                "total_frames": 0}

    probs      = [r["fake_prob"] for r in frame_results]
    mean_prob  = float(np.mean(probs))
    fake_votes = sum(1 for r in frame_results if r["pred"] == "FAKE")

    return {
        "verdict"        : LABEL_MAP[int(mean_prob >= threshold)],
        "mean_fake_prob" : round(mean_prob, 4),
        "frame_fake_votes" : fake_votes,
        "total_frames"   : len(frame_results),
    }


# ── Per-input wrappers ────────────────────────────────────────────────────────

def run_video(
    video_path: str,
    model: nn.Module,
    mtcnn,
    transform: transforms.Compose,
    device: torch.device,
    cfg: Config,
    verbose: bool = True,
) -> dict:
    frames = extract_frames(video_path, cfg.FRAMES_PER_VIDEO, mtcnn, cfg.FACE_SIZE)

    if not frames:
        print(f"  [WARN] No frames extracted from {video_path}")
        return {"file": video_path, "verdict": "UNKNOWN", "mean_fake_prob": None,
                "frame_results": []}

    frame_results = predict_frames(frames, model, transform, device, cfg.BATCH_SIZE)
    verdict       = aggregate_verdict(frame_results)

    result = {
        "file"           : video_path,
        "verdict"        : verdict["verdict"],
        "mean_fake_prob" : verdict["mean_fake_prob"],
        "frame_fake_votes" : verdict["frame_fake_votes"],
        "total_frames"   : verdict["total_frames"],
        "frame_results"  : frame_results,
    }

    if verbose:
        print(
            f"  {Path(video_path).name:<40s} | "
            f"{result['verdict']:4s} | "
            f"prob={result['mean_fake_prob']:.4f} | "
            f"fake_frames={result['frame_fake_votes']}/{result['total_frames']}"
        )

    return result


def run_image(
    image_path: str,
    model: nn.Module,
    mtcnn,
    transform: transforms.Compose,
    device: torch.device,
    cfg: Config,
) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    crop      = crop_face(frame_rgb, mtcnn, cfg.FACE_SIZE)
    results   = predict_frames([(0, crop)], model, transform, device, batch_size=1)
    r         = results[0]

    print(
        f"  {Path(image_path).name:<40s} | "
        f"{r['pred']:4s} | prob={r['fake_prob']:.4f}"
    )

    return {"file": image_path, "verdict": r["pred"],
            "mean_fake_prob": r["fake_prob"], "frame_results": results}


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Deepfake detection inference")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video",      type=str, help="Path to a single video file")
    src.add_argument("--video_dir",  type=str, help="Directory of video files (.mp4/.avi/.mov)")
    src.add_argument("--image",      type=str, help="Path to a single face image")
    src.add_argument("--manifest",   type=str, help="CSV with a 'path' column (images)")

    p.add_argument("--checkpoint",   type=str, default="checkpoints/best.pth",
                   help="Path to trained checkpoint (default: checkpoints/best.pth)")
    p.add_argument("--threshold",    type=float, default=0.5,
                   help="Fake probability threshold (default: 0.5)")
    p.add_argument("--batch_size",   type=int,   default=8,
                   help="Frame batch size for GPU inference (default: 8)")
    p.add_argument("--output",       type=str,   default=None,
                   help="Optional path to save results as JSON")
    p.add_argument("--no_mtcnn",     action="store_true",
                   help="Skip MTCNN; use centre-crop only (faster, less accurate)")
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = Config()
    cfg.BATCH_SIZE = args.batch_size   # allow CLI override

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Threshold  : {args.threshold}")

    # ── Load model ────────────────────────────────────────────────────────────
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            f"Train first with: python train.py"
        )
    model = load_model(args.checkpoint, cfg, device)

    # ── Load MTCNN ────────────────────────────────────────────────────────────
    if args.no_mtcnn:
        mtcnn = None
        # Monkey-patch crop_face to always use centre-crop
        global crop_face
        _orig_crop = crop_face
        def crop_face(frame_rgb, mtcnn, face_size):  # noqa: F811
            h, w   = frame_rgb.shape[:2]
            s      = min(h, w)
            y0, x0 = (h - s) // 2, (w - s) // 2
            return cv2.resize(frame_rgb[y0:y0+s, x0:x0+s], (face_size, face_size))
    else:
        mtcnn = get_mtcnn(cfg.FACE_SIZE, device)

    transform = get_inference_transform(cfg.FACE_SIZE)
    all_results = []

    # ── Single video ──────────────────────────────────────────────────────────
    if args.video:
        print(f"\nInferring: {args.video}")
        result = run_video(args.video, model, mtcnn, transform, device, cfg)
        all_results.append(result)

    # ── Video directory ───────────────────────────────────────────────────────
    elif args.video_dir:
        exts   = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        videos = sorted(
            p for p in Path(args.video_dir).iterdir()
            if p.suffix.lower() in exts
        )
        if not videos:
            raise FileNotFoundError(f"No video files found in {args.video_dir}")

        print(f"\nInferring {len(videos)} videos from {args.video_dir}\n")
        for vp in tqdm(videos, desc="Videos", unit="video"):
            result = run_video(str(vp), model, mtcnn, transform, device, cfg, verbose=True)
            all_results.append(result)

        # Summary
        verdicts  = [r["verdict"] for r in all_results if r["verdict"] != "UNKNOWN"]
        n_fake    = verdicts.count("FAKE")
        n_real    = verdicts.count("REAL")
        print(f"\nSummary — REAL: {n_real} | FAKE: {n_fake} | UNKNOWN: "
              f"{len(all_results) - len(verdicts)}")

    # ── Single image ──────────────────────────────────────────────────────────
    elif args.image:
        print(f"\nInferring: {args.image}")
        result = run_image(args.image, model, mtcnn, transform, device, cfg)
        all_results.append(result)

    # ── Manifest CSV (image paths) ────────────────────────────────────────────
    elif args.manifest:
        df = pd.read_csv(args.manifest)
        if "path" not in df.columns:
            raise ValueError("manifest CSV must contain a 'path' column")

        print(f"\nInferring {len(df)} images from manifest: {args.manifest}\n")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Images", unit="img"):
            try:
                result = run_image(row["path"], model, mtcnn, transform, device, cfg)
            except FileNotFoundError as e:
                print(f"  [WARN] {e}")
                result = {"file": row["path"], "verdict": "ERROR",
                          "mean_fake_prob": None, "frame_results": []}
            all_results.append(result)

    # ── Persist results ───────────────────────────────────────────────────────
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
