import argparse
import sys
import os
import urllib.request
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor

# MediaPipe Logic
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_ID       = "prithivMLmods/Deep-Fake-Detector-v2-Model"
DEEPFAKE_LABEL = "Deepfake"
REAL_LABEL     = "Realism"

VIT_MEAN: np.ndarray = None  
VIT_STD:  np.ndarray = None  

MIN_FACE_PX     = 60    
BLUR_THRESHOLD  = 80.0  
UNCERTAINTY_BAND = 0.08

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ── Model & Detector Setup ────────────────────────────────────────────────────

def get_mediapipe_model():
    """Ensure the TFLite model exists in temp storage."""
    model_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    model_path = os.path.join(tempfile.gettempdir(), "blaze_face_short_range.tflite")
    if not os.path.exists(model_path):
        print(f"[*] Downloading MediaPipe model...")
        urllib.request.urlretrieve(model_url, model_path)
    return model_path

def load_resources(device: torch.device):
    global VIT_MEAN, VIT_STD

    print(f"[*] Loading model  : {MODEL_ID}")
    processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    model     = ViTForImageClassification.from_pretrained(MODEL_ID)
    model.to(device).eval()

    VIT_MEAN = np.array(processor.image_mean, dtype=np.float32)
    VIT_STD  = np.array(processor.image_std,  dtype=np.float32)
    
    detector = None
    if _MEDIAPIPE_AVAILABLE:
        model_path = get_mediapipe_model()
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.6)
        detector = mp_vision.FaceDetector.create_from_options(options)
        print("[*] Face detector  : MediaPipe Tasks API")
    else:
        print("[*] Face detector  : Haar cascade (fallback)")
        detector = cv2.CascadeClassifier(CASCADE_PATH)

    return model, detector

# ── Detection Logic ───────────────────────────────────────────────────────────

def detect_faces(bgr: np.ndarray, detector) -> list[tuple]:
    H, W = bgr.shape[:2]
    
    if _MEDIAPIPE_AVAILABLE:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        
        boxes = []
        if result.detections:
            for det in result.detections:
                bb = det.bounding_box
                # MediaPipe Tasks returns absolute pixels for bounding_box
                x, y, w, h = int(bb.origin_x), int(bb.origin_y), int(bb.width), int(bb.height)
                if w >= MIN_FACE_PX and h >= MIN_FACE_PX:
                    boxes.append((x, y, w, h))
        return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(MIN_FACE_PX, MIN_FACE_PX))
        return sorted([tuple(f) for f in faces], key=lambda b: b[2] * b[3], reverse=True)

# ── Image Processing ──────────────────────────────────────────────────────────

def crop_face(bgr: np.ndarray, bbox: tuple, margin: float = 0.30) -> np.ndarray | None:
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

def build_tta_variants(pil_rgb: Image.Image) -> list[np.ndarray]:
    variants = [
        pil_rgb,
        ImageEnhance.Brightness(pil_rgb).enhance(1.10),
        ImageEnhance.Brightness(pil_rgb).enhance(0.90),
        ImageEnhance.Contrast(pil_rgb).enhance(1.10),
        pil_rgb.filter(ImageFilter.UnsharpMask(radius=1, percent=30, threshold=3)),
    ]
    outputs = []
    for v in variants:
        img = v.resize((224, 224), Image.BICUBIC)
        arr = (np.array(img, dtype=np.float32) / 255.0 - VIT_MEAN) / VIT_STD
        outputs.append(arr)
    return outputs

# ── Inference Pipeline ────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--margin", type=float, default=0.30)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    model, detector = load_resources(device)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, num=min(args.frames, total), dtype=int)
    
    results = []
    for idx in tqdm(indices, desc="Processing"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue

        faces = detect_faces(frame, detector)
        if not faces: continue

        crop = crop_face(frame, faces[0], margin=args.margin)
        if crop is None or is_blurry(crop): continue

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tta = build_tta_variants(pil)
        
        tensor = torch.from_numpy(np.stack(tta)).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            outputs = model(tensor).logits
            probs = F.softmax(outputs, dim=-1).mean(dim=0)

        results.append(probs[model.config.label2id[DEEPFAKE_LABEL]].item())

    cap.release()

    if not results:
        sys.exit("[!] No valid faces found.")

    # Temporal Smoothing & Aggregation
    smoothed = np.convolve(results, np.ones(3)/3, mode='valid') if len(results) > 2 else results
    mean_prob = np.mean(smoothed)
    verdict = DEEPFAKE_LABEL if mean_prob >= args.threshold else REAL_LABEL

    print(f"\n{'='*30}\nVERDICT: {verdict}\nCONFIDENCE: {max(mean_prob, 1-mean_prob)*100:.2f}%\n{'='*30}")

if __name__ == "__main__":
    main()
