"""Stage 1 — data loading, cleaning and schema normalisation.

Responsibilities
----------------
* Load the CICIDS-2017 ``*-no-metadata.parquet`` shards.
* Normalise raw labels to the canonical :class:`~backend.ml.taxonomy.AttackClass`.
* Drop constant / zero-variance columns (the shards each carry 10 dead columns).
* Repair ``±inf`` values produced by rate features with zero denominators
  (e.g. ``Flow Bytes/s`` when ``Flow Duration == 0``).
* Derive the two modelling targets:
    - ``y_binary``  — benign vs attack   (Level-1 classifier)
    - ``y_family``  — attack families    (Level-2 meta-classifier)

Everything here is a pure function of its inputs so it can be unit-tested and
re-run deterministically.  Model fitting (scalers, imputers) happens in
:mod:`backend.ml.pipeline` — this module only cleans.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .taxonomy import AttackClass

LABEL_COL = "Label"

# Columns that are identifiers/metadata rather than behavioural features.
# The "-no-metadata" exports have already removed source/destination IPs, but
# we keep the list explicit so a future metadata export fails loudly.
DROPPED_METADATA = [
    "Source IP",
    "Destination IP",
    "Source Port",
    "Destination Port",
    "Timestamp",
    "Flow ID",
]

# Rate/volume features are extremely heavy-tailed (values up to ~1e9).  A log1p
# transform keeps them inside a sane numeric range for the scalers and the
# gradient-boosted learners.
LOG1P_HINTS = (
    "bytes/s",
    "packets/s",
    "duration",
    "iat",
    "total",
    "header length",
)


def load_shards(
    paths: list[Path],
    *,
    verify_labels: bool = True,
) -> DatasetBundle:
    """Load and clean one or more parquet shards.

    Parameters
    ----------
    paths
        Parquet files to concatenate.  They must share a column schema.
    verify_labels
        Raise on unrecognised labels instead of silently dropping them.
    """
    if not paths:
        raise ValueError("load_shards() requires at least one parquet path")

    frames = []
    per_file = []
    for path in paths:
        path = Path(path)
        frame = pd.read_parquet(path)
        per_file.append({"file": path.name, "rows": int(len(frame))})
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    return clean_frame(raw, per_file=per_file, verify_labels=verify_labels)


def clean_frame(
    df: pd.DataFrame,
    *,
    per_file: list[dict] | None = None,
    verify_labels: bool = True,
) -> DatasetBundle:
    """Clean an already-loaded DataFrame and build modelling targets."""
    if LABEL_COL not in df.columns:
        raise KeyError(
            f"Expected a {LABEL_COL!r} column, got {list(df.columns)[:5]}..."
        )

    df = df.copy()
    raw_labels = df[LABEL_COL].astype(str).to_numpy()

    # --- labels -> canonical integer classes -------------------------------
    mapped = np.empty(len(df), dtype=np.int16)
    for i, label in enumerate(raw_labels):
        cls = _map_label(label, verify_labels=verify_labels)
        mapped[i] = -1 if cls is None else int(cls)

    keep = mapped >= 0
    df = df.loc[keep].reset_index(drop=True)
    raw_labels = raw_labels[keep]
    mapped = mapped[keep]

    y_family_all = mapped.astype(np.int64)
    y_binary = (y_family_all != int(AttackClass.BENIGN)).astype(np.int64)

    # --- features ----------------------------------------------------------
    features = df.drop(columns=[LABEL_COL])
    dropped_meta = [c for c in DROPPED_METADATA if c in features.columns]
    if dropped_meta:
        features = features.drop(columns=dropped_meta)

    # Numeric-only: any residual object column is not a flow statistic.
    non_numeric = features.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        features = features.drop(columns=non_numeric)

    # --- inf repair (rate features with a zero denominator) ----------------
    numeric = features.to_numpy(dtype=np.float64)
    inf_count = int(np.isinf(numeric).sum())
    numeric[np.isinf(numeric)] = np.nan  # parked; filled with the median below
    features = pd.DataFrame(numeric, columns=features.columns, index=features.index)

    # --- drop constant / near-constant columns -----------------------------
    nunique = features.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        features = features.drop(columns=constant_cols)

    # --- median imputation (covers the NaN we just parked) -----------------
    missing_before = int(features.isna().sum().sum())
    features = features.fillna(features.median(numeric_only=True))
    # A column that was entirely NaN would still be NaN -> fill with 0.
    features = features.fillna(0.0)

    report = {
        "files": per_file or [],
        "rows_total": int(len(df)),
        "rows_dropped_unlabelled": int((~keep).sum()),
        "dropped_metadata_columns": dropped_meta,
        "dropped_non_numeric_columns": non_numeric,
        "constant_columns_dropped": constant_cols,
        "inf_values_repaired": inf_count,
        "missing_values_imputed": missing_before,
        "columns_after_cleaning": int(features.shape[1]),
        "class_counts": {
            AttackClass(int(c)).name: int((y_family_all == c).sum())
            for c in sorted(set(y_family_all.tolist()))
        },
        "attack_rate": float(y_binary.mean()),
    }

    return DatasetBundle(
        X=features,
        y_binary=y_binary,
        y_family=y_family_all,
        raw_labels=raw_labels,
        feature_names=list(features.columns),
        report=report,
    )


def apply_log1p(X: pd.DataFrame, *, columns: list[str] | None = None) -> pd.DataFrame:
    """Sign-preserving ``log1p`` on heavy-tailed rate/volume features.

    Uses ``sign(x) * log1p(|x|)`` so negative inter-arrival-time deltas (which
    legitimately occur in the CICIDS feature set) are handled without producing
    NaN.
    """
    if columns is None:
        columns = [c for c in X.columns if any(h in c.lower() for h in LOG1P_HINTS)]

    out = X.copy()
    for col in columns:
        if col in out.columns:
            values = out[col].to_numpy(dtype=np.float64)
            out[col] = np.sign(values) * np.log1p(np.abs(values))
    return out


def _map_label(label: str, *, verify_labels: bool) -> AttackClass | None:
    """Resolve a raw label, optionally tolerating unknown families."""
    from .taxonomy import to_class

    try:
        return to_class(label)
    except KeyError:
        if verify_labels:
            raise
        return None


@dataclass(slots=True)
class DatasetBundle:
    """Clean, aligned feature matrix plus modelling targets."""

    X: pd.DataFrame
    y_binary: np.ndarray
    y_family: np.ndarray
    raw_labels: np.ndarray = field(default_factory=lambda: np.array([]))
    feature_names: list[str] = field(default_factory=list)
    report: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.feature_names:
            self.feature_names = list(self.X.columns)
