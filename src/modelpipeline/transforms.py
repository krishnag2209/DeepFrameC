"""
transforms.py — Augmentation pipeline for deepfake detection.

Train augmentations are designed specifically for FF++ deepfake artifacts:
  - JPEG compression noise (deepfakes are often re-compressed)
  - Gaussian blur (face region boundaries are often blurred)
  - Downscale+upscale (simulates lossy codec degradation)
  - RandomGraying + HSV jitter (color inconsistencies at blend boundaries)
  - GridDistortion (catches models that rely on rigid face geometry)
  - CoarseDropout (forces attention to multiple regions, not just center)
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(split: str, face_size: int = 380):
    if split == "train":
        return A.Compose([
            # ── Geometry ──────────────────────────────────────────────────────
            A.RandomResizedCrop(size=(face_size, face_size), scale=(0.80, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.4),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),

            # ── Compression / frequency artifacts ────────────────────────────
            # Deepfakes are almost always re-encoded at some point
            A.OneOf([
                A.ImageCompression(quality_range=(40, 85), p=1.0),
                A.Downscale(scale_range=(0.5, 0.9), p=1.0),
            ], p=0.6),

            # ── Blur / noise ──────────────────────────────────────────────────
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=7),
                A.GaussNoise(noise_scale_factor=0.15),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
            ], p=0.5),

            # ── Color ─────────────────────────────────────────────────────────
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08, p=0.6),
            A.ToGray(p=0.05),                  # rare grayscale, improves robustness
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                                 val_shift_limit=20, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),

            # ── Regularization ────────────────────────────────────────────────
            A.CoarseDropout(
                num_holes_range=(1, 6),
                hole_height_range=(20, 60),
                hole_width_range=(20, 60),
                p=0.4,
            ),

            # ── Normalize + tensor ────────────────────────────────────────────
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        # Val/test: deterministic — only resize and normalize
        return A.Compose([
            A.Resize(height=face_size, width=face_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
