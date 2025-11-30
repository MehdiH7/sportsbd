from __future__ import annotations

from typing import List

import torch
from PIL import Image

from sportsbd.inference import predict_clip


class DummyModel(torch.nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x shape: (N, C, T, H, W)
        batch_size = x.shape[0]
        # Return zeros logits so softmax is uniform.
        return torch.zeros(batch_size, self.num_classes, device=x.device)


def _make_dummy_frames(t: int = 16) -> List[Image.Image]:
    frames: List[Image.Image] = []
    for _ in range(t):
        frames.append(Image.new("RGB", (160, 120), color=(128, 128, 128)))
    return frames


def test_predict_clip_shapes() -> None:
    frames = _make_dummy_frames(16)
    model = DummyModel(num_classes=4)
    result = predict_clip(frames, model=model, device="cpu")

    assert "class_probs" in result
    assert "any_boundary_prob" in result
    assert len(result["class_probs"]) == 4
    assert 0.0 <= result["any_boundary_prob"] <= 1.0


