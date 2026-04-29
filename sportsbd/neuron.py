from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .config import DEFAULT_CONFIG


def trace_neuron_model(model: nn.Module, config: Dict[str, Any]) -> nn.Module:
    try:
        import torch_neuronx
    except ImportError as exc:
        raise RuntimeError(
            "torch_neuronx is required for the experimental 'neuron' backend, "
            "but it is not installed."
        ) from exc

    t_frames = int(config.get("T_FRAMES", 16))
    img_size = int(config.get("IMG_SIZE", DEFAULT_CONFIG.image_size))
    example_input = torch.randn(1, 3, t_frames, img_size, img_size)
    traced_model = torch_neuronx.trace(model, example_input)
    return traced_model
