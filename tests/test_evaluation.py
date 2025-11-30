from __future__ import annotations

from sportsbd.evaluation import Annotation, EvaluationResult, evaluate_predictions


def test_time_based_evaluation_basic() -> None:
    # Simple 1-1 matching within tolerance.
    preds = [
        {"timestamp_ms": 1000},
        {"timestamp_ms": 3000},
    ]
    gts = [
        Annotation(timestamp_ms=950),
        Annotation(timestamp_ms=3100),
    ]

    result = evaluate_predictions(preds, gts, tolerance_ms=200)
    assert isinstance(result, EvaluationResult)
    assert result.tp == 2
    assert result.fp == 0
    assert result.fn == 0
    assert result.f1_score == 1.0


