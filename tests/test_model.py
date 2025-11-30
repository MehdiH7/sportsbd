from __future__ import annotations

from pathlib import Path

import torch

from sportsbd.model import load_model


def test_load_model_from_checkpoint(tmp_path: Path) -> None:
    # Create a dummy r2plus1d_18-compatible state dict via the library itself.
    from torchvision.models.video import r2plus1d_18

    base_model = r2plus1d_18(weights=None)
    state_dict = base_model.state_dict()

    ckpt = {
        "state_dict": state_dict,
        "config": {
            "MODEL_NAME": "r2plus1d_18",
            "NUM_CLASSES": 4,
        },
    }

    ckpt_path = tmp_path / "model.ckpt"
    torch.save(ckpt, ckpt_path)

    model = load_model(ckpt_path, device="cpu")
    assert hasattr(model, "fc")
    assert model.fc.out_features == 4  # type: ignore[attr-defined]


