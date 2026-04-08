import os

class Config:
    DATA_ROOT  = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "FaceForensics")
    MANIPULATION_TYPES = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    COMPRESSION = "c23"
    FRAMES_PER_VIDEO = 30          # was 10 — more frames = richer signal per video
    FRAMES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "FaceForensics_transformed")
    FACE_SIZE  = 380               # EfficientNet-B4 native resolution

    TRAIN_SPLIT = "train"
    VAL_SPLIT   = "val"
    TEST_SPLIT  = "test"

    BATCH_SIZE       = 12          # smaller batch to fit 380px + dual-stream on VRAM
    GRAD_ACCUM_STEPS = 4           # effective batch = 12 * 4 = 48
    NUM_WORKERS      = 8
    EPOCHS           = 30          # staged: warmup(5) + full(25)
    WARMUP_EPOCHS    = 5           # backbone frozen for first 5 epochs
    LR               = 2e-4       # head/FPN/SRM LR; backbone gets LR/10 after warmup
    WEIGHT_DECAY     = 1e-4
    LABEL_SMOOTHING  = 0.1
    MIXUP_ALPHA      = 0.4         # beta distribution param for Mixup
    EMA_DECAY        = 0.9999      # shadow model decay

    PRETRAINED  = True
    DROPOUT     = 0.4
    NUM_CLASSES = 2

    CHECKPOINT_DIR = "checkpoints/"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
