"""Stage 4 — hybrid feature selection.

The corpus has 74 candidate features after Stages 2-3, many of which are
redundant (CICIDS features are highly collinear by construction — e.g.
``Total Fwd Packets`` and ``Subflow Fwd Packets`` are perfectly correlated).

Redundancy is not merely untidy here: it inflates inference cost on every
single flow and destabilises SHAP attributions, since collinear features split
credit arbitrarily between them.  Selection therefore matters for
*explainability* as much as for speed.

Three filters, cheapest and most aggressive first:

1. **Pearson** — drop one of any pair with ``|r|`` above the threshold.  Uses a
   greedy transitive-reduction walk, not naive pairwise pruning, so a chain of
   correlated features collapses to a single representative rather than being
   left partially connected.
2. **ANOVA F-test** — rank survivors by linear class separability.
3. **Chi-square** — rank by non-linear dependence on discretised labels.  Kept
   alongside ANOVA because a monotonic-only filter would drop U-shaped
   relationships entirely.
4. **GA wrapper** — a genetic search over the shortlist using PR-AUC as fitness,
   which finds feature *interactions* no univariate filter can see.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import MinMaxScaler

DEFAULT_PEARSON_THRESHOLD = 0.90
DEFAULT_SHORTLIST = 30


@dataclass(slots=True)
class SelectionResult:
    """Outcome of the hybrid selection pipeline."""

    selected: list[str]
    pearson_dropped: list[str]
    pearson_pairs: list[tuple[str, str, float]]
    anova_scores: dict[str, float]
    chi2_scores: dict[str, float]
    ga_fitness: float | None = None
    ga_generations: int = 0

    def summary(self) -> dict:
        return {
            "n_selected": len(self.selected),
            "selected": self.selected,
            "n_pearson_dropped": len(self.pearson_dropped),
            "top_pearson_pairs": self.pearson_pairs[:10],
            "ga_fitness": self.ga_fitness,
            "ga_generations": self.ga_generations,
        }


def pearson_prune(
    X: pd.DataFrame,
    *,
    threshold: float = DEFAULT_PEARSON_THRESHOLD,
    protected: set[str] | None = None,
) -> tuple[list[str], list[str], list[tuple[str, str, float]]]:
    """Greedily drop features that are collinear with an already-kept one.

    Iterating over the *original* column order and comparing each candidate only
    against features already accepted guarantees the result does not depend on
    dictionary/set iteration order, which is essential for reproducibility.

    Returns ``(kept, dropped, high_pairs)``.
    """
    corr = X.corr(numeric_only=True).abs()
    protected = protected or set()

    kept: list[str] = []
    dropped: list[str] = []
    high_pairs: list[tuple[str, str, float]] = []

    for col in X.columns:
        if col in protected:
            kept.append(col)
            continue

        duplicate_of = None
        worst = 0.0
        for anchor in kept:
            if anchor not in corr.columns or anchor in protected:
                continue
            value = corr.loc[col, anchor]
            if pd.isna(value):
                continue
            if value >= threshold and value > worst:
                worst = value
                duplicate_of = anchor

        if duplicate_of is None:
            kept.append(col)
        else:
            dropped.append(col)
            high_pairs.append((col, duplicate_of, float(worst)))

    high_pairs.sort(key=lambda t: t[2], reverse=True)
    return kept, dropped, high_pairs


def univariate_scores(
    X: pd.DataFrame,
    y: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    """ANOVA F-test and chi-square scores per feature.

    Chi-square needs non-negative input, so the features are min-max scaled
    into ``[0, 1]`` first.  That is a monotonic transform per column and does
    not change which features rank highly.
    """
    values = X.to_numpy(dtype=np.float64)

    f_scores, _ = f_classif(values, y)
    f_map = {
        col: (0.0 if np.isnan(score) else float(score))
        for col, score in zip(X.columns, f_scores)
    }

    scaled = MinMaxScaler().fit_transform(values)
    chi2_scores, _ = chi2(scaled, y)
    chi2_map = {
        col: (0.0 if np.isnan(score) else float(score))
        for col, score in zip(X.columns, chi2_scores)
    }

    return f_map, chi2_map


def shortlist_by_rank(
    anova: dict[str, float],
    chi2_scores: dict[str, float],
    *,
    k: int = DEFAULT_SHORTLIST,
    anova_weight: float = 0.5,
) -> list[str]:
    """Top-``k`` features by blended rank of both univariate scores.

    Averaging *ranks* rather than raw scores keeps the two measures on a common
    scale — F-tests and chi-square statistics have incomparable magnitudes, so
    averaging raw values would let ANOVA dominate purely by scale.
    """
    columns = list(anova.keys())

    anova_rank = _rank_desc(anova)
    chi2_rank = _rank_desc(chi2_scores)

    blended = {
        col: anova_weight * anova_rank[col] + (1 - anova_weight) * chi2_rank[col]
        for col in columns
    }
    # Lower blended rank score == better on both measures.
    return sorted(columns, key=lambda c: blended[c])[:k]


def _rank_desc(scores: dict[str, float]) -> dict[str, float]:
    """Rank features from best (0) to worst, as normalised floats."""
    ordered = sorted(scores, key=lambda c: scores[c], reverse=True)
    n = max(len(ordered) - 1, 1)
    return {col: i / n for i, col in enumerate(ordered)}


def hybrid_select(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    pearson_threshold: float = DEFAULT_PEARSON_THRESHOLD,
    shortlist_size: int = DEFAULT_SHORTLIST,
    protected: set[str] | None = None,
) -> SelectionResult:
    """Run Pearson -> ANOVA + chi-square shortlisting (no GA).

    The GA wrapper lives in :mod:`backend.ml.genetic` and is optional, because
    it is expensive; this function gives the deterministic filter baseline.
    """
    kept, dropped, pairs = pearson_prune(
        X, threshold=pearson_threshold, protected=protected
    )
    survivors = X[kept]

    anova, chi2_scores = univariate_scores(survivors, y)
    shortlist = shortlist_by_rank(anova, chi2_scores, k=shortlist_size)

    return SelectionResult(
        selected=shortlist,
        pearson_dropped=dropped,
        pearson_pairs=pairs,
        anova_scores=anova,
        chi2_scores=chi2_scores,
    )

    high_pairs.sort(key=lambda t: t[2], reverse=True)
    return kept, dropped, high_pairs
