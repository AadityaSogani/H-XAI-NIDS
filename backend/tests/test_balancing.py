"""Unit tests for k-Means SMOTE balancing."""
from __future__ import annotations

import numpy as np
import pytest

from ml.balancing import KMeansSMOTE


def make_blobs(n_majority: int = 900, n_minority: int = 30):
    """Two classes with a 30:1 imbalance, mirroring the real corpus."""
    rng = np.random.default_rng(5)
    majority = rng.normal(0.0, 1.0, size=(n_majority, 6))
    # Two well-separated minority sub-clusters, as in the real attack families.
    sub_a = rng.normal(6.0, 0.5, size=(n_minority // 2, 6))
    sub_b = rng.normal(-6.0, 0.5, size=(n_minority - n_minority // 2, 6))
    minority = np.vstack([sub_a, sub_b])

    X = np.vstack([majority, minority])
    y = np.concatenate([np.zeros(n_majority), np.ones(n_minority)]).astype(int)
    return X, y


def test_rejects_invalid_ratio():
    with pytest.raises(ValueError):
        KMeansSMOTE(ratio=0.0)
    with pytest.raises(ValueError):
        KMeansSMOTE(ratio=1.5)


def test_minority_is_upsampled():
    X, y = make_blobs()
    X_bal, y_bal = KMeansSMOTE(ratio=0.3, random_state=0).fit_resample(X, y)

    before = int((y == 1).sum())
    after = int((y_bal == 1).sum())

    assert after > before
    assert after >= int(0.3 * (y == 0).sum())


def test_majority_is_never_touched():
    """Oversampling must add rows, never remove or duplicate the majority."""
    X, y = make_blobs()
    X_bal, y_bal = KMeansSMOTE(ratio=0.3).fit_resample(X, y)

    assert int((y_bal == 0).sum()) == int((y == 0).sum())


def test_output_is_finite():
    X, y = make_blobs()
    X_bal, _ = KMeansSMOTE(ratio=0.3).fit_resample(X, y)

    assert np.isfinite(X_bal).all()


def test_synthetic_rows_stay_near_real_minority():
    """k-Means SMOTE must not invent points in the empty space between blobs.

    The two minority sub-clusters sit at +6 and -6; the benign class sits at 0.
    If neighbours were chosen globally, interpolation would fill the gap around
    0 and place synthetic attacks on top of the majority.
    """
    X, y = make_blobs(n_minority=40)
    X_bal, y_bal = KMeansSMOTE(ratio=0.3, n_clusters=4, random_state=0).fit_resample(X, y)

    synth = X_bal[len(X):][y_bal[len(X):] == 1]

    # Distance from each synthetic point to the nearest benign centroid (0).
    dist_to_benign = np.linalg.norm(synth - 0.0, axis=1)
    assert (dist_to_benign > 2.0).all(), (
        "synthetic attacks were generated inside the benign region"
    )


def test_synthetic_rows_are_not_exact_duplicates():
    X, y = make_blobs(n_minority=200)
    X_bal, _ = KMeansSMOTE(ratio=0.3, random_state=0).fit_resample(X, y)

    synth = X_bal[len(X):]
    assert len(np.unique(synth, axis=0)) > len(synth) * 0.9


def test_tiny_class_falls_back_to_jitter():
    """SQL_INJECTION has 21 rows corpus-wide: too few to interpolate."""
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 1, size=(500, 5)), rng.normal(8, 0.4, size=(6, 5))])
    y = np.concatenate([np.zeros(500), np.ones(6)]).astype(int)

    X_bal, y_bal = KMeansSMOTE(ratio=0.3, k_neighbors=5).fit_resample(X, y)

    assert np.isfinite(X_bal).all()
    assert (y_bal == 1).sum() > 6


def test_single_class_is_untouched():
    X = np.random.default_rng(2).normal(size=(50, 4))
    y = np.zeros(50, dtype=int)

    X_bal, y_bal = KMeansSMOTE().fit_resample(X, y)

    assert X_bal.shape == X.shape
    assert np.array_equal(y_bal, y)


def test_multiclass_generates_for_every_minority():
    rng = np.random.default_rng(3)
    X = np.vstack(
        [
            rng.normal(0, 1, size=(600, 4)),
            rng.normal(6, 0.5, size=(40, 4)),
            rng.normal(-6, 0.5, size=(20, 4)),
        ]
    )
    y = np.concatenate([np.zeros(600), np.ones(40), np.full(20, 2)]).astype(int)

    result = KMeansSMOTE(ratio=0.3, random_state=0)
    X_bal, y_bal = result.fit_resample(X, y)

    assert (y_bal == 1).sum() > 40
    assert (y_bal == 2).sum() > 20
    assert set(result.describe()["synthetic_per_class"]) == {1, 2}


def test_describe_reports_generation_counts():
    X, y = make_blobs()
    result = KMeansSMOTE(ratio=0.3)
    result.fit_resample(X, y)
    summary = result.describe()

    assert summary["ratio"] == 0.3
    assert 1 in summary["synthetic_per_class"]
    assert summary["synthetic_per_class"][1] > 0


def test_is_deterministic():
    X, y = make_blobs(n_minority=60)

    a = KMeansSMOTE(ratio=0.3, random_state=7).fit_resample(X, y)
    b = KMeansSMOTE(ratio=0.3, random_state=7).fit_resample(X, y)

    assert a[0].shape == b[0].shape
    assert np.allclose(a[0], b[0])
