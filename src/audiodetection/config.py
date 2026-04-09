import os

class Config:
    # ── Data ──────────────────────────────────────────────────────────────────
    DATA_ROOT   = "data/WaveFake"
    SAMPLE_RATE = 16000
    DURATION    = 4                          # seconds
    MAX_SAMPLES = SAMPLE_RATE * DURATION     # 64000 samples

    # ── Mel spectrogram ───────────────────────────────────────────────────────
    N_MELS     = 128
    N_FFT      = 1024
    HOP_LENGTH = 256                         # was 512 → finer time resolution
    TOP_DB     = 80.0                        # dynamic range clamp for AmplitudeToDB

    # ── Splits ────────────────────────────────────────────────────────────────
    TRAIN_SPLIT = "train"
    VAL_SPLIT   = "val"
    TEST_SPLIT  = "test"

    # ── Training ──────────────────────────────────────────────────────────────
    BATCH_SIZE      = 32
    NUM_WORKERS     = 8
    EPOCHS          = 30
    LR              = 2e-4
    WEIGHT_DECAY    = 1e-4
    LABEL_SMOOTHING = 0.05

    # ── Model ─────────────────────────────────────────────────────────────────
    PRETRAINED  = True
    DROPOUT     = 0.3
    NUM_CLASSES = 2

    # ── Checkpoint ────────────────────────────────────────────────────────────
    CHECKPOINT_DIR = "checkpoints"
