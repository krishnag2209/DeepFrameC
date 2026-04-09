import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(split: str, face_size: int = 224):
        return A.Compose([
            A.Resize(height=face_size, width=face_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
