from __future__ import annotations

from typing import Callable

import torch
from PIL import Image
from torchvision import transforms as T

from .config import DEFAULT_CONFIG


def _pil_to_tensor() -> T.Compose:
    return T.Compose(
        [
            T.Resize((DEFAULT_CONFIG.image_size, DEFAULT_CONFIG.image_size)),
            T.ToTensor(),
            T.Normalize(mean=DEFAULT_CONFIG.mean, std=DEFAULT_CONFIG.std),
        ]
    )


def get_frame_transform() -> Callable[[Image.Image], torch.Tensor]:
    """
    Return a transform that converts a PIL Image into a normalized tensor (C, H, W).
    """
    return _pil_to_tensor()


def normalize_tensor_frame(frame: torch.Tensor) -> torch.Tensor:
    """
    Normalize a tensor frame (C, H, W) using Kinetics-style mean/std.
    """
    if frame.ndim != 3:
        raise ValueError(f"Expected frame tensor with 3 dims (C, H, W), got shape {tuple(frame.shape)}")

    mean = torch.tensor(DEFAULT_CONFIG.mean, dtype=frame.dtype, device=frame.device)[:, None, None]
    std = torch.tensor(DEFAULT_CONFIG.std, dtype=frame.dtype, device=frame.device)[:, None, None]
    return (frame - mean) / std


