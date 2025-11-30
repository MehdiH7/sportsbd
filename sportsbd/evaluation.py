from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class Annotation:
    timestamp_ms: int
    label: Optional[str] = None


@dataclass
class EvaluationResult:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1_score: float


def _extract_prediction_timestamps(predictions: Sequence[Dict[str, Any]]) -> List[int]:
    timestamps: List[int] = []
    for pred in predictions:
        if not isinstance(pred, dict):
            continue
        if "timestamp_ms" not in pred:
            continue
        timestamps.append(int(pred["timestamp_ms"]))
    return timestamps


def _greedy_time_matching(
    pred_times: List[int],
    gt_times: List[int],
    tolerance_ms: int,
) -> EvaluationResult:
    pred_times_sorted = sorted(pred_times)
    gt_times_sorted = sorted(gt_times)

    used_gt = [False] * len(gt_times_sorted)
    tp = 0
    fp = 0

    for pt in pred_times_sorted:
        best_idx = -1
        best_dist = None
        for i, gt in enumerate(gt_times_sorted):
            if used_gt[i]:
                continue
            dist = abs(pt - gt)
            if dist <= tolerance_ms and (best_dist is None or dist < best_dist):
                best_idx = i
                best_dist = dist
        if best_idx >= 0:
            used_gt[best_idx] = True
            tp += 1
        else:
            fp += 1

    fn = used_gt.count(False)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return EvaluationResult(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
    )


def evaluate_predictions(
    predictions: Sequence[Dict[str, Any]],
    ground_truth: Sequence[Annotation],
    tolerance_frames: Optional[int] = None,
    tolerance_ms: Optional[int] = None,
    fps: int = 25,
) -> EvaluationResult:
    """
    Evaluate predictions vs. ground truth using time-based matching.

    Exactly one of tolerance_frames or tolerance_ms must be provided.
    """
    if (tolerance_frames is None) == (tolerance_ms is None):
        raise ValueError("Exactly one of tolerance_frames or tolerance_ms must be provided.")

    if tolerance_ms is None:
        tolerance_ms = int(tolerance_frames * 1000 / fps)  # type: ignore[arg-type]

    pred_times = _extract_prediction_timestamps(predictions)
    gt_times = [ann.timestamp_ms for ann in ground_truth]

    return _greedy_time_matching(pred_times, gt_times, tolerance_ms=tolerance_ms)


