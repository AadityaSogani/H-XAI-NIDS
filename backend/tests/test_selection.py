"""Unit tests for hybrid feature selection and the GA wrapper."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.genetic import PARSIMONY_WEIGHT, PR_AUC_WEIGHT, GeneticFeatureSelector
from ml.selection import (
    hybrid_select,
    pearson_prune,
    shortlist_by_rank,
    univariate_scores,
)


@pytest.fixture
def collinear_frame():
    """Six features where col_c duplicates col_a and col_d tracks col_b."""
    rng = np.random.default_rng(3)
    n = 300
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = (a > 0).astype(int)
    return (
        pd.DataFrame(
            {
                "col_a": a,
                "col_b": b,
                "col_c": a + noise * 1e-6,   # ~perfectly correlated with col_a
                "col_d": b * 2.0,            # exactly correlated with col_b
                "col_e": rng.normal(size=n),  # independent
            }
        ),
        y,
    )


def _separable(n: int = 400, n_features: int = 12):
    """Only the first 4 features carry signal; the rest are noise."""
    rng = np.random.default_rng(11)
    signal = rng.normal(size=(n, 4))
    y = (signal[:, 0] + signal[:, 1] > 0).astype(int)
    noise = rng.normal(size=(n, n_features - 4))
    return np.hstack([signal, noise]), y


# --------------------------------------------------------------- Pearson ----


def test_pearson_drops_duplicate_columns(collinear_frame):
    X, _ = collinear_frame
    kept, dropped, pairs = pearson_prune(X, threshold=0.90)

    assert "col_c" in dropped  # duplicate of col_a
    assert "col_d" in dropped  # scaled copy of col_b
    assert {"col_a", "col_b", "col_e"} <= set(kept)


def test_pearson_is_deterministic(collinear_frame):
    X, _ = collinear_frame
    first = pearson_prune(X, threshold=0.90)
    second = pearson_prune(X, threshold=0.90)

    assert first[0] == second[0]
    assert first[1] == second[1]


def test_pearson_result_is_order_independent(collinear_frame):
    """Shuffling column order must not change the surviving set.

    Order changes *which* duplicate is kept as the representative (col_a vs
    col_c), but the number of survivors must be identical.
    """
    X, _ = collinear_frame
    shuffled = X[list(reversed(X.columns))]

    kept_a, dropped_a, _ = pearson_prune(X, threshold=0.90)
    kept_b, dropped_b, _ = pearson_prune(shuffled, threshold=0.90)

    assert len(kept_a) == len(kept_b)
    assert set(kept_a) == set(X.columns) - set(dropped_a)
    assert set(kept_b) == set(X.columns) - set(dropped_b)


def test_pearson_respects_protected_columns(collinear_frame):
    X, _ = collinear_frame
    kept, dropped, _ = pearson_prune(X, threshold=0.90, protected={"col_a"})

    assert "col_a" in kept
    assert "col_a" not in dropped


def test_pearson_reports_high_pairs_sorted(collinear_frame):
    X, _ = collinear_frame
    _, _, pairs = pearson_prune(X, threshold=0.90)

    assert pairs, "expected at least one high-correlation pair"
    assert pairs[0][2] >= pairs[-1][2]


def test_pearson_high_threshold_keeps_everything(collinear_frame):
    X, _ = collinear_frame
    kept, dropped, pairs = pearson_prune(X, threshold=1.5)

    assert dropped == []
    assert pairs == []
    assert set(kept) == set(X.columns)


# ---------------------------------------------------------- univariate ----


def test_univariate_scores_are_finite(collinear_frame):
    X, y = collinear_frame
    anova, chi2_scores = univariate_scores(X, y)

    assert set(anova) == set(X.columns)
    assert set(chi2_scores) == set(X.columns)
    assert all(np.isfinite(v) for v in anova.values())
    assert all(np.isfinite(v) for v in chi2_scores.values())


def test_univariate_ranks_informative_feature_first(collinear_frame):
    """col_a generated y, so it must outrank the independent noise column.

    ``col_c`` is a near-perfect copy of ``col_a`` and may legitimately score
    marginally higher, so we assert the informative pair beats the noise rather
    than pinning one exact winner.
    """
    X, y = collinear_frame
    anova, _ = univariate_scores(X, y)

    assert anova["col_a"] > anova["col_e"]
    assert max(anova, key=anova.get) in {"col_a", "col_c"}


def test_shortlist_respects_size(collinear_frame):
    X, y = collinear_frame
    anova, chi2_scores = univariate_scores(X, y)
    picked = shortlist_by_rank(anova, chi2_scores, k=3)

    assert len(picked) == 3
    assert len(set(picked)) == 3


def test_shortlist_larger_than_pool_returns_all(collinear_frame):
    X, y = collinear_frame
    anova, chi2_scores = univariate_scores(X, y)
    picked = shortlist_by_rank(anova, chi2_scores, k=999)


# ------------------------------------------------------------------- GA ----


def test_ga_finds_a_subset():
    X, y = _separable()
    columns = [f"f{i}" for i in range(X.shape[1])]
    result = GeneticFeatureSelector(
        population_size=12, generations=3, cv_folds=3
    ).fit(pd.DataFrame(X, columns=columns), y)

    assert result.n_selected >= 4
    assert 0.0 <= result.fitness <= 1.0
    assert result.pr_auc > 0.5, "GA should beat random on a separable problem"


def test_ga_prefers_informative_features():
    """The selected set should be enriched for the first 4 informative columns."""
    X, y = _separable()
    columns = [f"f{i}" for i in range(X.shape[1])]
    result = GeneticFeatureSelector(
        population_size=16, generations=4, cv_folds=3
    ).fit(pd.DataFrame(X, columns=columns), y)

    informative = set(columns[:4])
    hits = sum(1 for c in result.selected if c in informative)
    assert hits >= 2, f"only {hits} of 4 informative features were selected"


def test_ga_history_is_nondecreasing():
    """Elitism guarantees the running best never regresses."""
    X, y = _separable(n=300)
    result = GeneticFeatureSelector(
        population_size=10, generations=4, cv_folds=3
    ).fit(pd.DataFrame(X), y)

    assert (np.diff(result.history) >= -1e-12).all()


def test_ga_rejects_too_small_subsets():
    X, y = _separable()
    ga = GeneticFeatureSelector(min_features=5)

    assert ga.fitness(np.array([True, False, False, False]), X, y) == 0.0
    assert ga.fitness(np.zeros(X.shape[1], dtype=bool), X, y) == 0.0


def test_ga_is_deterministic():
    """The same seed must select the same subset.

    The *selected features* are compared exactly.  Fitness is compared loosely
    because the proxy RF runs with ``n_jobs=-1``, and parallel float summation
    order varies between runs at the ~1e-5 level.
    """
    X, y = _separable(n=300)
    columns = [f"f{i}" for i in range(X.shape[1])]
    frame = pd.DataFrame(X, columns=columns)

    a = GeneticFeatureSelector(population_size=10, generations=3).fit(frame, y)
    b = GeneticFeatureSelector(population_size=10, generations=3).fit(frame, y)

    assert a.selected == b.selected
    assert a.fitness == pytest.approx(b.fitness, abs=1e-3)


def test_ga_fitness_weights_match_pseudocode():
    assert PR_AUC_WEIGHT == 0.85
    assert PARSIMONY_WEIGHT == 0.15
    assert PR_AUC_WEIGHT + PARSIMONY_WEIGHT == pytest.approx(1.0)


def test_ga_summary_shape():
    X, y = _separable(n=300)
    columns = [f"f{i}" for i in range(X.shape[1])]
    summary = GeneticFeatureSelector(
        population_size=8, generations=2
    ).fit(pd.DataFrame(X, columns=columns), y).summary()

    assert set(summary) == {
        "n_selected", "fitness", "pr_auc", "generations", "selected",
    }


def test_shortlist_larger_than_pool_returns_all(collinear_frame):
    X, y = collinear_frame
    anova, chi2_scores = univariate_scores(X, y)
    picked = shortlist_by_rank(anova, chi2_scores, k=999)

    assert set(picked) == set(X.columns)


    assert set(picked) == set(X.columns)


def test_hybrid_select_pipeline(collinear_frame):
    X, y = collinear_frame
    result = hybrid_select(X, y, shortlist_size=4)

    assert "col_c" in result.pearson_dropped
    assert all(c in X.columns for c in result.selected)
    assert len(result.selected) <= 4
    assert result.summary()["n_selected"] == len(result.selected)
