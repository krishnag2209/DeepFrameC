import os

class Config:
    DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "FaceForensics")
    MANIPULATION_TYPES = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    COMPRESSION = "c23"
    FRAMES_PER_VIDEO = 10
    FRAMES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "FaceForensics_transformed")
    FACE_SIZE = 224

    TRAIN_SPLIT = "train"
    VAL_SPLIT   = "val"
    TEST_SPLIT  = "test"

    BATCH_SIZE  = 16         # Reduced to fit larger images in VRAM
    NUM_WORKERS = 10
    EPOCHS      = 20         # More epochs for convergence
    LR          = 1e-4       # Higher LR; 3e-5 was too slow to escape random chance
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.1

    BACKBONE    = "vit_base_patch16_224"
    PRETRAINED  = True
    DROPOUT     = 0.5
    NUM_CLASSES = 2

    CHECKPOINT_DIR = "checkpoints/"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
