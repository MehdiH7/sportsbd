"""
Device detection utilities for automatic GPU/CPU selection.
"""

from __future__ import annotations

import torch


def get_available_device(prefer: str | None = None) -> torch.device:
    """
    Automatically detect and return the best available device.
    
    Priority order: cuda > mps > cpu.
    
    Args:
        prefer: Optional device preference ("cuda", "mps", "cpu", or None for auto).
    
    Returns:
        torch.device: The best available device.
    """
    # Respect explicit user preference first
    if prefer is not None:
        prefer = prefer.lower()
        if prefer == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if prefer == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if prefer == "cpu":
            return torch.device("cpu")
        # If preferred device is not available, fall through to auto-detection.
    
    # Auto-detect: try CUDA first
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # Then try MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    
    # Fall back to CPU
    return torch.device("cpu")


def get_device_name(device: torch.device | str) -> str:
    """
    Get a human-readable name for a device.
    
    Args:
        device: torch.device or device string
    
    Returns:
        str: Human-readable device name
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    if device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    elif device.type == "mps":
        return "MPS (Apple Silicon)"
    elif device.type == "cpu":
        return "CPU"
    else:
        return str(device)

