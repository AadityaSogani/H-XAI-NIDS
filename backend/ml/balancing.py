"""Stage 5 — cluster-aware class balancing (k-Means SMOTE).

Why not plain SMOTE?
--------------------
Standard SMOTE interpolates between a minority sample and one of its *global*
k-nearest neighbours.  When a minority class forms several well-separated
sub-clusters, global neighbours can straddle the empty space *between* them, so
SMOTE invents synthetic points in regions that contain no real attack at all.
A classifier then learns a boundary that does not exist.  At a 1.23% attack
rate this is not a rounding error — it is the difference between a usable
detector and one that fires on noise.

k-Means SMOTE (Last et al., 2017) fixes this by searching for neighbours
*within a cluster only*, and by generating more synthetic samples in clusters
where the minority is scarcest.  Those are precisely the regions a flat
classifier handles worst.

Classes too small to interpolate (``SQL_INJECTION`` has 21 rows corpus-wide)
fall back to random oversampling with small Gaussian jitter, because SMOTE
needs at least ``k_neighbors + 1`` members to have anything to interpolate
between.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


class KMeansSMOTE:
    """Cluster-aware oversampler for heavily imbalanced tabular data.

    Parameters
    ----------
    ratio
        Target minority size as a fraction of the majority class size.  ``0.3``
        upsamples attacks to 30% of the benign count — enough to stop the
        learner ignoring the minority, without flooding the training set with
        synthetic rows.
    n_clusters, k_neighbors
        Clusters to search within, and neighbours per interpolation.
    """

    def __init__(
        self,
        *,
        ratio: float = 0.3,
        n_clusters: int = 8,
        k_neighbors: int = 5,
        jitter: float = 0.05,
        random_state: int = 42,
    ) -> None:
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        self.ratio = ratio
        self.n_clusters = n_clusters
        self.k_neighbors = k_neighbors
        self.jitter = jitter
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the original rows plus synthetic minority rows."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)
        if classes.size < 2:
            self.generated_per_class_: dict[int, int] = {}
            return X.copy(), y.copy()

        majority = int(counts.max())
        target = max(int(self.ratio * majority), majority + 1)

        rng = np.random.default_rng(self.random_state)
        count_of = dict(zip(classes.tolist(), counts.tolist()))

        # Cluster the MINORITY rows only: we want the sub-structure of the rare
        # classes, not centroids dragged toward the 99% benign majority.
        minority_mask = counts[y] < majority
        X_min, y_min = X[minority_mask], y[minority_mask]

        n_clusters = int(min(self.n_clusters, max(2, len(X_min))))
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=self.random_state)
        cluster_of = kmeans.fit_predict(X_min)

        new_X: list[np.ndarray] = []
        new_y: list[np.ndarray] = []
        generated: dict[int, int] = {}

        for cluster_id in range(n_clusters):
            in_cluster = cluster_of == cluster_id
            if not in_cluster.any():
                continue

            X_c, y_c = X_min[in_cluster], y_min[in_cluster]

            for cls in np.unique(y_c):
                cls = int(cls)
                deficit = target - count_of[cls]
                if deficit <= 0:
                    continue

                # Spread this class's shortfall across clusters in proportion to
                # how many of its rows live in each one, so sparse sub-clusters
                # receive more synthetic support.
                idx = np.flatnonzero(y_c == cls)
                share = len(idx) / max(int((y_min == cls).sum()), 1)
                n_to_make = int(round(deficit * share))
                if n_to_make <= 0:
                    continue

                if len(idx) >= self.k_neighbors + 1:
                    synth = self._smote_in_cluster(X_c[idx], n_to_make, rng)
                else:
                    # Too few members to interpolate -> jittered duplication.
                    synth = self._jitter(X_c[idx], n_to_make, rng)

                new_X.append(synth)
                new_y.append(np.full(len(synth), cls, dtype=y.dtype))
                generated[cls] = generated.get(cls, 0) + len(synth)

        self.generated_per_class_ = generated
        X_bal = np.vstack([X, *new_X]) if new_X else X
        y_bal = np.concatenate([y, *new_y]) if new_y else y
        return X_bal, y_bal

    def _smote_in_cluster(
        self, rows: np.ndarray, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Interpolate between same-cluster, same-class neighbours only."""
        diff = rows[:, None, :] - rows[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)

        k = int(min(self.k_neighbors, len(rows) - 1))
        neighbours = np.argsort(dist, axis=1)[:, :k]

        base = rng.integers(0, len(rows), size=n)
        picks = neighbours[base, rng.integers(0, k, size=n)]
        gaps = rng.random(n)[:, None]

        return rows[base] + gaps * (rows[picks] - rows[base])

    def _jitter(
        self, rows: np.ndarray, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Duplicate rows with small relative noise (avoids exact copies)."""
        picks = rng.integers(0, len(rows), size=n)
        base = rows[picks]
        scale = np.abs(base).mean(axis=0, keepdims=True) + 1e-9
        return base + rng.normal(0.0, self.jitter, size=base.shape) * scale

    def describe(self) -> dict:
        """What was generated, for the evaluation report."""
        return {
            "ratio": self.ratio,
            "n_clusters": self.n_clusters,
            "k_neighbors": self.k_neighbors,
            "synthetic_per_class": getattr(self, "generated_per_class_", {}),
        }
