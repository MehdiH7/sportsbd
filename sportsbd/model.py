from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18

from .config import DEFAULT_CONFIG


def _build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name != "r2plus1d_18":
        raise ValueError(f"Unsupported model_name={model_name!r}, only 'r2plus1d_18' is supported.")

    model = r2plus1d_18(weights=None)
    in_features = model.fc.in_features  # type: ignore[attr-defined]
    model.fc = nn.Linear(in_features, num_classes)  # type: ignore[attr-defined]
    return model


def _load_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=map_location)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        config = checkpoint.get("config", {})
    elif isinstance(checkpoint, dict):
        # Assume this is a bare state_dict
        state_dict = checkpoint
        config = {}
    else:
        raise ValueError("Checkpoint must be a dict containing a 'state_dict' or be a state_dict itself.")

    if not isinstance(config, dict):
        raise ValueError("Checkpoint 'config' must be a dict if provided.")

    return state_dict, config


def load_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cuda",
) -> nn.Module:
    """
    Load an r2plus1d_18 model from a checkpoint.

    The checkpoint is expected to contain at least:
      - 'state_dict': model weights
      - 'config': dict with MODEL_NAME and NUM_CLASSES (optional; defaults applied)
    """
    map_location = torch.device(device) if isinstance(device, str) else device
    state_dict, config = _load_checkpoint(checkpoint_path, map_location=map_location)

    model_name = config.get("MODEL_NAME", DEFAULT_CONFIG.model_name)
    num_classes = int(config.get("NUM_CLASSES", DEFAULT_CONFIG.num_classes))

    model = _build_model(model_name=model_name, num_classes=num_classes)

    # Handle potential prefixes like "module."
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module.") :]] = v
        else:
            new_state_dict[k] = v

    # Filter out incompatible keys (e.g., fc layer when num_classes differs)
    model_state_dict = model.state_dict()
    compatible_state_dict = {}
    for k, v in new_state_dict.items():
        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape:
                compatible_state_dict[k] = v
            else:
                # Skip keys with shape mismatches (e.g., fc layer)
                print(f"[sportsbd] Skipping incompatible key '{k}': checkpoint shape {v.shape} != model shape {model_state_dict[k].shape}")
        else:
            # Key not in model, skip it
            pass

    missing, unexpected = model.load_state_dict(compatible_state_dict, strict=False)
    if missing:
        # Not raising: warn via print to avoid breaking usage; users can inspect.
        print(f"[sportsbd] Warning: missing keys in state_dict: {missing}")
    if unexpected:
        print(f"[sportsbd] Warning: unexpected keys in state_dict: {unexpected}")

    model.to(map_location)
    model.eval()
    return model


