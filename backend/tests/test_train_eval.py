"""Unit tests for the evaluation block ml.train persists into the report.

The dashboard charts per-class F1, the confusion matrix and the PR/ROC curves
from training_report.json, so the shape and naming of this block is a public
contract.  These tests run on tiny synthetic arrays — no dataset needed.
"""
from __future__ import annotations

import numpy as np

from ml.train import _evaluate


def _verdicts(y_pred: np.ndarray) -> list[dict]:
    return [{"label_id": int(p)} for p in y_pred]


def test_evaluate_labels_are_canonical_and_aligned() -> None:
    rng = np.random.default_rng(0)
    n = 300
    y_true = rng.choice([0, 1, 3], size=n, p=[0.7, 0.2, 0.1]).astype(int)
    y_bin = (y_true != 0).astype(int)
    y_pred = y_true.copy()
    y_pred[0] = -1  # one Level-3 escalation
    p_attack = np.clip(rng.normal(0.5, 0.25, n), 0.0, 1.0)

    result = _evaluate(y_true, y_bin, p_attack, _verdicts(y_pred))

    labels = result["confusion_matrix"]["labels"]
    # Canonical taxonomy names (matching serving verdict labels) + UNSEEN.
    assert labels == ["BENIGN", "BOT", "BRUTE_FORCE", "UNSEEN"]
    assert set(result["per_class"]) == set(labels)


def test_confusion_matrix_shape_matches_labels() -> None:
    rng = np.random.default_rng(1)
    n = 120
    y_true = rng.choice([0, 2], size=n).astype(int)
    y_bin = (y_true != 0).astype(int)
    p_attack = rng.uniform(size=n)

    result = _evaluate(y_true, y_bin, p_attack, _verdicts(y_true))
    cm = result["confusion_matrix"]

    k = len(cm["labels"])
    assert len(cm["matrix"]) == k
    assert all(len(row) == k for row in cm["matrix"])
    # Every row sums to the true-class support.
    for i, row in enumerate(cm["matrix"]):
        assert sum(row) == int((y_true == (0 if cm["labels"][i] == "BENIGN" else 2)).sum())


def test_curves_are_aligned_and_bounded() -> None:
    rng = np.random.default_rng(2)
    n = 200
    y_true = (rng.uniform(size=n) > 0.9).astype(int)
    p_attack = rng.uniform(size=n)

    curves = _evaluate(y_true, y_true, p_attack, _verdicts(y_true))["curves"]

    for key in ("pr", "roc"):
        arrays = curves[key]
        lengths = {len(v) for v in arrays.values()}
        assert len(lengths) == 1, f"{key} curve axes differ in length"
        for values in arrays.values():
            assert all(0.0 <= v <= 1.0 for v in values)

    # The payload must stay small: the trainer downsamples to ~100 points.
    assert len(curves["roc"]["fpr"]) <= 101


def test_per_class_metrics_have_precision_recall_f1_support() -> None:
    rng = np.random.default_rng(3)
    n = 100
    y_true = np.zeros(n, dtype=int)
    y_true[:10] = 3

    result = _evaluate(y_true, (y_true != 0).astype(int), rng.uniform(size=n), _verdicts(y_true))

    for metrics in result["per_class"].values():
        assert set(metrics) == {"precision", "recall", "f1", "support"}
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
        assert metrics["support"] >= 0
