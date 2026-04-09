"""
Deepfake Video Detection - Inference Script
Model : prithivMLmods/Deep-Fake-Detector-v2-Model  (ViT-base, ~92% accuracy)
Labels: "Realism" (real) | "Deepfake" (fake)

FIXES APPLIED
─────────────
1. Dynamic normalisation: reads actual mean/std from ViTImageProcessor instead
   of hardcoding [0.5,0.5,0.5] — the #1 cause of the 50/50 confidence bug.
2. MediaPipe face detector replaces unreliable Haar cascade; falls back to
   Haar automatically if mediapipe is not installed.
3. Uncertainty guard: flags results where mean deepfake prob is within 8 % of
   0.5, i.e. the model genuinely cannot decide.
4. Face crop margin: includes forehead/chin/ear boundary artefacts (key for deepfakes).
5. Blur/quality gate: skips frames where the face crop is too blurry or small.
6. Test-Time Augmentation (TTA): 5 mild photometric variants per face crop;
   softmax probabilities are averaged → reduces single-frame prediction noise.
7. Temporal smoothing: median filter over the frame sequence before aggregation.
8. Configurable decision threshold (default 0.50, tune with --threshold).

Usage:
    python inference.py --video path/to/video.mp4
    python inference.py --video path/to/video.mp4 --frames 80 --device cuda --verbose
    python inference.py --video path/to/video.mp4 --no_tta          # faster, less accurate
    python inference.py --video path/to/video.mp4 --threshold 0.55  # stricter deepfake gate

Requirements (core):
    pip install torch torchvision transformers opencv-python Pillow tqdm numpy

Optional (better face detection):
    pip install mediapipe
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor

# Optional mediapipe — detected at runtime
try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_ID       = "prithivMLmods/Deep-Fake-Detector-v2-Model"
DEEPFAKE_LABEL = "Deepfake"
REAL_LABEL     = "Realism"

# Populated dynamically from the processor after load_model() is called.
# Do NOT hardcode these — wrong values are the #1 cause of 50/50 outputs.
VIT_MEAN: np.ndarray = None  # type: ignore[assignment]
VIT_STD:  np.ndarray = None  # type: ignore[assignment]

# Face quality thresholds
MIN_FACE_PX     = 60    # minimum face bounding-box side in pixels
BLUR_THRESHOLD  = 80.0  # Laplacian variance; below = too blurry

# Uncertainty band: if |mean_deepfake_prob - 0.5| < this → warn user
UNCERTAINTY_BAND = 0.08

# OpenCV Haar cascade (fallback when mediapipe is unavailable)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(device: torch.device):
    """Load model and processor; extract real normalisation stats from processor."""
    global VIT_MEAN, VIT_STD

    print(f"[*] Loading model  : {MODEL_ID}")
    processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    model     = ViTForImageClassification.from_pretrained(MODEL_ID)
    model.to(device).eval()

    # ── FIX 1: use the processor's actual mean/std, not hardcoded [0.5,0.5,0.5] ──
    VIT_MEAN = np.array(processor.image_mean, dtype=np.float32)
    VIT_STD  = np.array(processor.image_std,  dtype=np.float32)
    print(f"[*] Normalisation  : mean={VIT_MEAN.tolist()}  std={VIT_STD.tolist()}")
    print(f"[*] Device         : {device}")
    print(f"[*] Face detector  : {'MediaPipe' if _MEDIAPIPE_AVAILABLE else 'Haar cascade (install mediapipe for better accuracy)'}")
    return processor, model


# ── Face detection — MediaPipe (primary) ──────────────────────────────────────

def detect_faces_mediapipe(bgr: np.ndarray) -> list[tuple]:
    """
    Detect faces with MediaPipe Face Detection (much more reliable than Haar).
    Returns list of (x, y, w, h) bounding boxes sorted largest-first.
    """
    mp_face = mp.solutions.face_detection
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = bgr.shape[:2]

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.60) as det:
        result = det.process(rgb)

    if not result.detections:
        return []

    boxes = []
    for detection in result.detections:
        bb = detection.location_data.relative_bounding_box
        x = max(0, int(bb.xmin * W))
        y = max(0, int(bb.ymin * H))
        w = int(bb.width  * W)
        h = int(bb.height * H)
        if w >= MIN_FACE_PX and h >= MIN_FACE_PX:
            boxes.append((x, y, w, h))

    return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


# ── Face detection — Haar cascade (fallback) ──────────────────────────────────

_cascade = None

def get_cascade():
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if _cascade.empty():
            sys.exit("[!] OpenCV Haar cascade not found. Reinstall opencv-python.")
    return _cascade


def detect_faces_haar(bgr: np.ndarray) -> list[tuple]:
    """Return bounding boxes (x,y,w,h), sorted largest-first."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade = get_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor  = 1.05,
        minNeighbors = 5,
        minSize      = (MIN_FACE_PX, MIN_FACE_PX),
        flags        = cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return []
    return sorted([tuple(f) for f in faces], key=lambda b: b[2] * b[3], reverse=True)


def detect_faces(bgr: np.ndarray) -> list[tuple]:
    """Dispatch to MediaPipe if available, otherwise Haar."""
    if _MEDIAPIPE_AVAILABLE:
        return detect_faces_mediapipe(bgr)
    return detect_faces_haar(bgr)


# ── Face crop & quality gate ──────────────────────────────────────────────────

def crop_face(bgr: np.ndarray, bbox: tuple, margin: float = 0.30) -> np.ndarray | None:
    """
    Crop face with proportional margin on all sides.
    The margin captures boundary artefacts (blending seams, etc.) that are
    key discriminative cues for deepfake detection.
    """
    H, W = bgr.shape[:2]
    x, y, w, h = bbox
    mx, my = int(w * margin), int(h * margin)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(W, x + w + mx), min(H, y + h + my)
    if (x2 - x1) < MIN_FACE_PX or (y2 - y1) < MIN_FACE_PX:
        return None
    return bgr[y1:y2, x1:x2]


def is_blurry(bgr_crop: np.ndarray) -> bool:
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD


# ── Pre-processing & TTA ──────────────────────────────────────────────────────

def preprocess(pil_img: Image.Image) -> np.ndarray:
    """
    Resize to 224×224 (bicubic, matching ViTImageProcessor default) and
    normalise with the processor's actual mean/std read at load time.
    """
    img = pil_img.resize((224, 224), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - VIT_MEAN) / VIT_STD          # shape (224, 224, 3)
    return arr


def build_tta_variants(pil_rgb: Image.Image) -> list[np.ndarray]:
    """
    5-view Test-Time Augmentation using mild photometric transforms.
    No horizontal flip — asymmetric blending/GAN artefacts carry signal.
    """
    variants = [
        pil_rgb,                                                   # 1. original
        ImageEnhance.Brightness(pil_rgb).enhance(1.10),           # 2. +10% brightness
        ImageEnhance.Brightness(pil_rgb).enhance(0.90),           # 3. -10% brightness
        ImageEnhance.Contrast(pil_rgb).enhance(1.10),             # 4. +10% contrast
        pil_rgb.filter(ImageFilter.UnsharpMask(radius=1,          # 5. light sharpening
                                               percent=30,
                                               threshold=3)),
    ]
    return [preprocess(v) for v in variants]


def arrays_to_tensor(arrays: list[np.ndarray], device: torch.device) -> torch.Tensor:
    """Stack HWC float32 arrays → NCHW tensor on device."""
    stacked = np.stack(arrays, axis=0)                       # (N, 224, 224, 3)
    tensor  = torch.from_numpy(stacked).permute(0, 3, 1, 2) # (N, 3, 224, 224)
    return tensor.to(device)


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(video_path: str, n_frames: int) -> list[np.ndarray]:
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
        if ret:
            frames.append(bgr)
    cap.release()
    return frames


# ── Per-frame inference ───────────────────────────────────────────────────────

def run_inference(
    frames: list[np.ndarray],
    model: ViTForImageClassification,
    device: torch.device,
    use_tta: bool,
    margin: float,
) -> list[dict]:
    """
    For each frame: detect face → quality gate → TTA → infer → record result.
    Frames with no valid face are silently skipped (counted and reported).
    """
    id2label     = model.config.id2label
    deepfake_idx = next(k for k, v in id2label.items() if v == DEEPFAKE_LABEL)
    real_idx     = next(k for k, v in id2label.items() if v == REAL_LABEL)

    results   = []
    n_no_face = 0
    n_blurry  = 0

    for frame_idx, bgr in enumerate(tqdm(frames, desc="Analysing faces", unit="frame")):
        faces = detect_faces(bgr)
        if not faces:
            n_no_face += 1
            continue

        crop = crop_face(bgr, faces[0], margin=margin)
        if crop is None:
            n_no_face += 1
            continue

        if is_blurry(crop):
            n_blurry += 1
            continue

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        arrays = build_tta_variants(pil) if use_tta else [preprocess(pil)]
        tensor = arrays_to_tensor(arrays, device)   # (N, 3, 224, 224)

        with torch.no_grad():
            logits = model(pixel_values=tensor).logits       # (N, C)
            probs  = F.softmax(logits, dim=-1).mean(dim=0)   # (C,)  — TTA average

        dp = probs[deepfake_idx].item()
        rp = probs[real_idx].item()

        results.append({
            "frame_idx":     frame_idx,
            "deepfake_prob": dp,
            "real_prob":     rp,
            "label":         DEEPFAKE_LABEL if dp > rp else REAL_LABEL,
        })

    print(f"[*] Valid faces    : {len(results)}")
    if n_no_face: print(f"[*] No face found  : {n_no_face} frames (skipped)")
    if n_blurry:  print(f"[*] Blurry/skipped : {n_blurry} frames")

    return results


# ── Temporal smoothing ────────────────────────────────────────────────────────

def temporal_smooth(per_frame: list[dict], window: int = 5) -> list[dict]:
    """Median-filter deepfake_prob over time to suppress single-frame noise."""
    if len(per_frame) < 3:
        return per_frame

    probs    = np.array([r["deepfake_prob"] for r in per_frame])
    pad      = window // 2
    padded   = np.pad(probs, pad, mode="edge")
    smoothed = np.array([np.median(padded[i:i + window]) for i in range(len(probs))])

    out = []
    for i, r in enumerate(per_frame):
        r2 = dict(r)
        r2["deepfake_prob"] = float(smoothed[i])
        r2["real_prob"]     = 1.0 - float(smoothed[i])
        r2["label"]         = DEEPFAKE_LABEL if smoothed[i] > 0.5 else REAL_LABEL
        out.append(r2)
    return out


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(per_frame: list[dict], threshold: float) -> dict:
    dps = [r["deepfake_prob"] for r in per_frame]
    rps = [r["real_prob"]     for r in per_frame]

    md = float(np.mean(dps))
    mr = float(np.mean(rps))

    verdict    = DEEPFAKE_LABEL if md >= threshold else REAL_LABEL
    confidence = max(md, mr) * 100.0
    n_fake     = sum(1 for r in per_frame if r["label"] == DEEPFAKE_LABEL)

    # ── FIX 3: flag low-confidence / uncertain results ──────────────────────
    is_uncertain = abs(md - 0.5) < UNCERTAINTY_BAND

    return {
        "verdict":            verdict,
        "confidence":         confidence,
        "mean_deepfake_prob": md * 100.0,
        "mean_real_prob":     mr * 100.0,
        "deepfake_frames":    n_fake,
        "total_frames":       len(per_frame),
        "deepfake_frame_pct": n_fake / len(per_frame) * 100.0,
        "threshold_used":     threshold * 100.0,
        "uncertain":          is_uncertain,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(agg: dict, per_frame: list[dict], verbose: bool):
    bar = "═" * 54
    print(f"\n{bar}")
    print(f"  VERDICT    : {agg['verdict'].upper()}")
    print(f"  CONFIDENCE : {agg['confidence']:.2f}%")
    print(f"  THRESHOLD  : {agg['threshold_used']:.1f}%")

    # ── FIX 3: uncertainty warning ──────────────────────────────────────────
    if agg["uncertain"]:
        print(f"\n  ⚠  LOW CONFIDENCE — result may be unreliable.")
        print(f"     Mean deepfake prob ({agg['mean_deepfake_prob']:.1f}%) is within")
        print(f"     {UNCERTAINTY_BAND*100:.0f}% of 50 % — the model is uncertain.")
        print(f"     Suggestions:")
        print(f"       • Use a higher-quality / higher-resolution video")
        print(f"       • Increase --frames (e.g. --frames 80)")
        print(f"       • Ensure the video contains clearly visible frontal faces")
        if not _MEDIAPIPE_AVAILABLE:
            print(f"       • pip install mediapipe  (better face detection)")

    print(f"\n{bar}")
    print(f"  Mean deepfake prob : {agg['mean_deepfake_prob']:.2f}%")
    print(f"  Mean real    prob  : {agg['mean_real_prob']:.2f}%")
    print(f"  Deepfake frames    : {agg['deepfake_frames']} / {agg['total_frames']}"
          f"  ({agg['deepfake_frame_pct']:.1f}%)")
    print(f"{bar}\n")

    if verbose:
        print("Per-frame breakdown (face-detected frames only):")
        print(f"  {'#':>5}  {'FrameIdx':>8}  {'Label':<12}  {'Deepfake%':>10}  {'Real%':>8}")
        print("  " + "-" * 50)
        for i, r in enumerate(per_frame):
            print(f"  {i+1:>5}  {r['frame_idx']:>8}  {r['label']:<12}  "
                  f"{r['deepfake_prob']*100:>9.2f}%  {r['real_prob']*100:>7.2f}%")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Deepfake video detection — face-aware, TTA-enhanced inference"
    )
    p.add_argument("--video",      required=True,
                   help="Path to input video file")
    p.add_argument("--frames",     type=int, default=40,
                   help="Frames to uniformly sample (default: 40)")
    p.add_argument("--device",     default="auto",
                   choices=["auto", "cpu", "cuda", "mps"],
                   help="Compute device (default: auto)")
    p.add_argument("--threshold",  type=float, default=0.50,
                   help="Deepfake decision threshold 0-1 (default: 0.50)")
    p.add_argument("--margin",     type=float, default=0.30,
                   help="Face crop margin as fraction of bbox (default: 0.30)")
    p.add_argument("--no_tta",     action="store_true",
                   help="Disable TTA — faster but less stable")
    p.add_argument("--verbose",    action="store_true",
                   help="Print per-frame probabilities")
    return p.parse_args()


def resolve_device(s: str) -> torch.device:
    if s == "auto":
        if torch.cuda.is_available():         return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(s)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = resolve_device(args.device)

    if not Path(args.video).is_file():
        sys.exit(f"[!] File not found: {args.video}")
    if not 0.0 < args.threshold < 1.0:
        sys.exit("[!] --threshold must be between 0 and 1")

    _, model  = load_model(device)
    frames    = extract_frames(args.video, args.frames)
    per_frame = run_inference(frames, model, device,
                              use_tta=not args.no_tta,
                              margin=args.margin)

    if not per_frame:
        print("\n[!] No valid face detections across all sampled frames.")
        print("    Suggestions:")
        print("    - Increase --frames (e.g. --frames 80)")
        print("    - Reduce --margin (e.g. --margin 0.10) to accept smaller crops")
        print("    - Ensure the video contains frontal human faces")
        if not _MEDIAPIPE_AVAILABLE:
            print("    - pip install mediapipe  (much better face detection)")
        sys.exit(2)

    per_frame = temporal_smooth(per_frame, window=5)
    agg       = aggregate(per_frame, threshold=args.threshold)
    print_report(agg, per_frame, args.verbose)

    sys.exit(1 if agg["verdict"] == DEEPFAKE_LABEL else 0)


if __name__ == "__main__":
    main()
