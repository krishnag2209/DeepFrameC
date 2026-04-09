import torchaudio.transforms as T
import torch.nn as nn


def get_audio_transforms(split: str) -> nn.Module:
    """
    SpecAugment for training; identity for val/test.
    Applied after log-mel + normalisation, so values are in [0, 1].
    """
    if split == "train":
        return nn.Sequential(
            T.FrequencyMasking(freq_mask_param=20),   # mask up to 20 mel bins
            T.TimeMasking(time_mask_param=40),         # mask up to 40 time frames
            T.TimeMasking(time_mask_param=40),         # second time mask (SpecAugment-2x)
        )
    return nn.Identity()
