from __future__ import annotations

from typing import List

import torch
from PIL import Image

from sportsbd.inference import predict_clip


class DummyNeuronModel(torch.nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.forward_called = False

    def to(self, *args: object, **kwargs: object) -> torch.nn.Module:  # type: ignore[override]
        raise AssertionError("Neuron traced models should not be moved with .to() during inference.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        self.forward_called = True
        assert x.device.type == "cpu"
        return torch.zeros(x.shape[0], self.num_classes)


def _make_dummy_frames(t: int = 16) -> List[Image.Image]:
    frames: List[Image.Image] = []
    for _ in range(t):
        frames.append(Image.new("RGB", (160, 120), color=(128, 128, 128)))
    return frames


def test_predict_clip_neuron_uses_cpu_tensors(monkeypatch) -> None:
    monkeypatch.setattr("sportsbd.inference.get_available_device", lambda prefer=None: "neuron")

    model = DummyNeuronModel()
    result = predict_clip(_make_dummy_frames(), model=model, device="neuron")

    assert model.forward_called
    assert len(result["class_probs"]) == 4
