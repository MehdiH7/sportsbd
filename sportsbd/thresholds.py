from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

from .evaluation import Annotation, EvaluationResult, evaluate_predictions


@dataclass
class ThresholdMetrics:
    threshold: float
    result: EvaluationResult


def sweep_thresholds(
    predictions: Sequence[Mapping],
    ground_truth: Sequence[Annotation],
    thresholds: Iterable[float],
    tolerance_frames: int | None = None,
    tolerance_ms: int | None = None,
    fps: int = 25,
) -> List[ThresholdMetrics]:
    """
    Sweep over confidence thresholds and compute evaluation metrics.

    Each prediction is expected to have a 'confidence' field.
    """
    thresholds_list = sorted(set(float(t) for t in thresholds))
    metrics: List[ThresholdMetrics] = []

    for thr in thresholds_list:
        filtered = [p for p in predictions if float(p.get("confidence", 0.0)) >= thr]
        result = evaluate_predictions(
            filtered,
            ground_truth,
            tolerance_frames=tolerance_frames,
            tolerance_ms=tolerance_ms,
            fps=fps,
        )
        metrics.append(ThresholdMetrics(threshold=thr, result=result))

    return metrics


