"""
Default configuration values for the sportsbd package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SportsBDConfig:
    model_name: str = "r2plus1d_18"
    num_classes: int = 4
    class_names: List[str] = None  # type: ignore[assignment]
    image_size: int = 112
    fps: int = 25

    # Kinetics-style normalization
    mean: tuple = (0.43216, 0.394666, 0.37645)
    std: tuple = (0.22803, 0.22145, 0.216989)


DEFAULT_CONFIG = SportsBDConfig(
    class_names=["hard", "fadein", "logo", "NaN"],
)


