"""Reproducible train / validation / test splitting.

Two constraints drive the design:

1. **Extreme class imbalance.**  The full corpus is 98.77% benign, and the
   rarest family (``SQL_INJECTION``) has only 21 rows across all three shards.
   Every split must therefore be *stratified on the family label*, otherwise a
   random split can easily leave a class absent from training entirely.

2. **No leakage between tiers.**  The Level-1 (binary) and Level-2 (family)
   classifiers must see identical row partitions, otherwise out-of-fold Level-1
   predictions used as Level-2 meta-features would be computed from rows the
   model has already seen.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


@dataclass(slots=True)
class Split:
    """Row-index partition of a dataset.  Indices refer to the parent arrays."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def __post_init__(self) -> None:
        for name in ("train", "validation", "test"):
            getattr(self, name).astype(np.int64, copy=False)

    def summary(self) -> dict[str, int]:
        return {
            "train": int(self.train.size),
            "validation": int(self.validation.size),
            "test": int(self.test.size),
        }

    def assert_disjoint(self) -> None:
        """Guarantee the three tiers share no row, and cover everything once."""
        train, val, test = self.train, self.validation, self.test
        assert len(set(train) & set(val)) == 0, "train/validation overlap"
        assert len(set(train) & set(test)) == 0, "train/test overlap"
        assert len(set(val) & set(test)) == 0, "validation/test overlap"


def stratified_split(
    y_family: np.ndarray,
    *,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: int = RANDOM_STATE,
    min_per_class: int = 3,
) -> Split:
    """Split indices stratified on the *family* label.

    Parameters
    ----------
    y_family
        Integer family labels (the most granular target).
    test_size, validation_size
        Fractions of the whole dataset.
    min_per_class
        Minimum rows a class must receive in *both* validation and test before
        it is stratified.  Classes below this are kept entirely in ``train`` —
        with 21 SQL-injection rows a 10% validation slice would leave ~2 rows,
        making any metric computed on it meaningless noise.
    """
    y_family = np.asarray(y_family)
    idx = np.arange(y_family.size)

    classes, counts = np.unique(y_family, return_counts=True)
    threshold = _rows_needed(test_size, validation_size, min_per_class)
    rare = classes[counts < threshold]

    rare_mask = np.isin(y_family, rare)
    idx_rare = idx[rare_mask]          # held out of stratification -> train only
    idx_common = idx[~rare_mask]

    if idx_common.size == 0 or np.unique(y_family[idx_common]).size < 2:
        # Degenerate: nothing left to stratify (no common rows, or a single
        # surviving class).  Everything goes to train.
        return Split(
            train=idx,
            validation=np.array([], dtype=np.int64),
            test=np.array([], dtype=np.int64),
        )

    y_common = y_family[idx_common]
    stratify = y_common if np.unique(y_common).size > 1 else None

    idx_trainval, idx_test = train_test_split(
        idx_common,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    val_fraction_of_train = validation_size / (1.0 - test_size)
    strat_trainval = None if stratify is None else y_family[idx_trainval]

    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_fraction_of_train,
        random_state=random_state,
        stratify=strat_trainval,
    )

    train = np.sort(np.concatenate([idx_train, idx_rare]))
    return Split(
        train=train,
        validation=np.sort(idx_val),
        test=np.sort(idx_test),
    )


def _rows_needed(test_size: float, validation_size: float, min_per_class: int) -> int:
    """Smallest class size that still yields ``min_per_class`` rows in BOTH tiers.

    A class is split twice: first ``test_size`` into test, then
    ``validation_size`` of the *original* into validation.  The validation slice
    is the tighter constraint, because it is taken from the smaller remaining
    pool (``1 - test_size``), so the required size is
    ``min_per_class / validation_size``.  We take the max of both constraints to
    stay safe if the fractions are ever reordered.
    """
    denom = max(min(test_size, validation_size), 1e-9)
    return int(np.ceil(min_per_class / denom))

