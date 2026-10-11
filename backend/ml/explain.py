"""Stage 7 — explainability (global TreeSHAP + local surrogate).

Explainability is not a cosmetic layer here.  Collinear features split SHAP
credit arbitrarily between them, so an unstable explanation sends an analyst
chasing the wrong column.  Two consequences shape this module:

* **Global** explanations come from TreeSHAP over the *balanced* training set,
  because that is the distribution the models were fitted on.  Explaining the
  raw corpus would describe a distribution the classifier never saw.
* **Local** explanations use a neighbour-based surrogate rather than LIME, for
  a specific reason: LIME perturbs tabular rows by sampling per-column
  statistics independently, which produces combinations that cannot occur in a
  real flow (negative packet counts, IAT smaller than the flow duration).  A
  surrogate fitted on those rows explains a neighbourhood the model will never
  be asked about.

We instead perturb along the segments between a row and its real near
neighbours, so every explained point is a physically plausible flow.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class FeatureAttribution:
    """One feature's contribution to a single prediction."""

    feature: str
    value: float
    contribution: float

    def as_dict(self) -> dict:
        return {
            "feature": self.feature,
            "value": round(float(self.value), 4),
            "contribution": round(float(self.contribution), 6),
        }


def _positive_class_shap(values: np.ndarray) -> np.ndarray:
    """Reduce SHAP output to the positive-class plane.

    Binary classifiers return shape ``(n, features, 2)``; the second class
    index is the attack probability, which is what an analyst cares about.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


class TreeShapExplainer:
    """Global and local explanations for a fitted tree ensemble via SHAP."""

    def __init__(
        self,
        model,
        feature_names: list[str],
        *,
        background: np.ndarray,
    ):
        self.model = model
        self.feature_names = list(feature_names)
        self.background = np.asarray(background, dtype=np.float64)
        self._explainer = None

    def _ensure_explainer(self):
        """Build the TreeExplainer lazily; it walks every tree."""
        if self._explainer is None:
            import shap

            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def global_importance(self, *, sample_size: int = 2000) -> list[dict]:
        """Mean |SHAP| per feature over a background sample.

        Mean *absolute* value is used rather than the mean signed value because
        positive and negative contributions cancel, hiding features that matter
        strongly in both directions.
        """
        X = _subsample(self.background, sample_size)
        values = _positive_class_shap(self._ensure_explainer().shap_values(X))

        mean_abs = np.abs(values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]

        return [
            {
                "rank": rank,
                "feature": self.feature_names[idx],
                "mean_abs_shap": round(float(mean_abs[idx]), 6),
            }
            for rank, idx in enumerate(order, start=1)
        ]

    def local_explanation(self, x: np.ndarray, *, top_k: int = 10) -> list[dict]:
        """Per-feature SHAP contributions for a single row."""
        row = np.asarray(x, dtype=np.float64).reshape(1, -1)
        values = _positive_class_shap(self._ensure_explainer().shap_values(row))
        contributions = values[0]

        order = np.argsort(np.abs(contributions))[::-1][:top_k]
        return [
            FeatureAttribution(
                feature=self.feature_names[i],
                value=float(row[0, i]),
                contribution=float(contributions[i]),
            ).as_dict()
            for i in order
        ]


class SurrogateExplainer:
    """Local explanation by fitting an interpretable model on a neighbourhood.

    ``r_squared`` is reported alongside the coefficients because a surrogate
    that does not reproduce the model locally is not an explanation of it.  A
    poor fit must be surfaced, not presented as insight.
    """

    def __init__(
        self,
        predict_fn,
        feature_names: list[str],
        *,
        n_samples: int = 500,
        n_neighbours: int = 25,
        random_state: int = 42,
    ):
        self.predict_fn = predict_fn
        self.feature_names = list(feature_names)
        self.n_samples = n_samples
        self.n_neighbours = n_neighbours
        self.random_state = random_state

    def explain(self, x: np.ndarray, background: np.ndarray) -> dict:
        """Fit a linear surrogate locally and return its coefficients."""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        row = np.asarray(x, dtype=np.float64).reshape(1, -1)
        rng = np.random.default_rng(self.random_state)

        neighbours = self._nearest(row, background)
        X_local = np.vstack([row, self._perturb(row, neighbours, rng)])

        scaler = StandardScaler().fit(X_local)
        X_scaled = scaler.transform(X_local)
        y_local = np.asarray(self.predict_fn(X_local), dtype=np.float64)

        surrogate = Ridge(alpha=1.0).fit(X_scaled, y_local)
        coefficients = surrogate.coef_
        order = np.argsort(np.abs(coefficients))[::-1]

        return {
            "fidelity_r2": round(float(surrogate.score(X_scaled, y_local)), 4),
            "intercept": round(float(surrogate.intercept_), 6),
            "contributions": [
                FeatureAttribution(
                    feature=self.feature_names[i],
                    value=float(row[0, i]),
                    contribution=float(coefficients[i]),
                ).as_dict()
                for i in order
            ],
        }

    def _nearest(self, row: np.ndarray, background: np.ndarray) -> np.ndarray:
        if len(background) == 0:
            return np.repeat(row, self.n_neighbours, axis=0)

        dist = np.sqrt(((background - row) ** 2).sum(axis=1))
        k = min(self.n_neighbours, len(background))
        return background[np.argsort(dist)[:k]]

    def _perturb(
        self, row: np.ndarray, neighbours: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample variations that stay on segments between real flows."""
        picks = rng.integers(0, len(neighbours), size=self.n_samples)
        base = neighbours[picks]
        gaps = rng.random((self.n_samples, 1))
        return row + gaps * (base - row)



def _subsample(X: np.ndarray, size: int, *, random_state: int = 42) -> np.ndarray:
    if len(X) <= size:
        return X
    rng = np.random.default_rng(random_state)
    return X[rng.choice(len(X), size=size, replace=False)]
