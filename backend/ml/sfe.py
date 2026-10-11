"""Stage 3 — Stacking Feature Embedding (SFE).

Tree ensembles treat every flow as an independent row.  They cannot see that a
flow sits deep inside a dense cluster of attacks, or on the fringe of benign
traffic — yet that *structural* position is often the strongest signal.

SFE recovers it with unsupervised learning and hands the result back to the
supervised models as nine extra columns:

* ``K-Means`` with ``k`` centroids -> 5 soft-distance features (how close the
  flow is to each centroid).  Soft distances preserve gradation; hard cluster
  IDs would throw away "nearly in cluster 3" information.
* ``Gaussian Mixture Model`` with ``m`` components -> 4 log-responsibility
  features (which latent regime the flow most likely belongs to).

Both are fitted on **training rows only**.  Fitting a scaler or clusterer on the
full corpus would leak validation/test distribution information into training.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

N_SFE_FEATURES = 9  # 5 K-Means distances + 4 GMM responsibilities


@dataclass(slots=True)
class SFETransformer:
    """Fit-once / transform-many unsupervised embedding."""

    n_clusters: int = 5
    n_components: int = 4
    random_state: int = 42

    scaler_: StandardScaler | None = None
    kmeans_: KMeans | None = None
    gmm_: GaussianMixture | None = None
    feature_names_: list[str] = field(default_factory=list)

    def fit(self, X: pd.DataFrame) -> "SFETransformer":
        """Learn the scaler, centroids and mixture from training data."""
        values = self._as_float(X)

        self.scaler_ = StandardScaler().fit(values)
        scaled = self.scaler_.transform(values)

        # Deterministic init + explicit n_init keeps results reproducible.
        n_clusters = min(self.n_clusters, max(2, len(X)))
        self.kmeans_ = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=self.random_state,
        ).fit(scaled)

        n_components = min(self.n_components, max(2, len(X)))
        self.gmm_ = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=self.random_state,
        ).fit(scaled)

        self.feature_names_ = [f"SFE_KM_Dist_{i}" for i in range(n_clusters)] + [
            f"SFE_GMM_Resp_{i}" for i in range(n_components)
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the 9 embedding columns for arbitrary rows."""
        if self.kmeans_ is None or self.gmm_ is None:
            raise RuntimeError("SFETransformer.transform() called before fit()")

        values = self._as_float(X)
        scaled = self.scaler_.transform(values)

        # Distances to every centroid -> gradient of "closeness".
        dists = self.kmeans_.transform(scaled)

        # log p(cluster | x) for each mixture component.
        resp = self.gmm_.predict_proba(scaled)
        log_resp = np.log(np.clip(resp, 1e-12, None))

        out = np.hstack([dists, log_resp])
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return pd.DataFrame(out, columns=self.feature_names_, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    @property
    def n_features_out(self) -> int:
        return len(self.feature_names_)

    @staticmethod
    def _as_float(X: pd.DataFrame) -> np.ndarray:
        return X.to_numpy(dtype=np.float64)
