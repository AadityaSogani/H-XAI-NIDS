"""Unit tests for Stage-1 cleaning (:mod:`ml.data`)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.data import DROPPED_METADATA, apply_log1p, clean_frame


def make_frame(n: int = 100, *, label: str = "Benign") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Flow Duration": rng.integers(1, 10_000, n),
            "Total Fwd Packets": rng.integers(1, 50, n),
            "Flow Bytes/s": rng.normal(1e6, 1e5, n),
            "Bwd PSH Flags": np.zeros(n, dtype=int),  # constant -> must be dropped
            "Label": label,
        }
    )


def test_constant_columns_dropped() -> None:
    bundle = clean_frame(make_frame())
    assert "Bwd PSH Flags" not in bundle.X.columns
    assert "Bwd PSH Flags" in bundle.report["constant_columns_dropped"]


def test_label_column_never_a_feature() -> None:
    bundle = clean_frame(make_frame())
    assert "Label" not in bundle.X.columns
    assert bundle.feature_names == list(bundle.X.columns)


def test_binary_target_marks_attacks() -> None:
    attack = clean_frame(make_frame(label="PortScan"))
    assert set(np.unique(attack.y_binary).tolist()) == {1}
    assert attack.report["attack_rate"] == 1.0


def test_benign_target_is_zero() -> None:
    benign = clean_frame(make_frame(label="Benign"))
    assert set(np.unique(benign.y_binary).tolist()) == {0}


def test_inf_values_are_repaired() -> None:
    """Rate features with a zero denominator produce ±inf; these must be fixed."""
    df = make_frame()
    df.loc[0, "Flow Bytes/s"] = np.inf
    df.loc[1, "Flow Bytes/s"] = -np.inf

    bundle = clean_frame(df)

    assert bundle.report["inf_values_repaired"] == 2
    assert not np.isinf(bundle.X.to_numpy()).any()
    assert not bundle.X.isna().any().any()


def test_nan_values_are_imputed() -> None:
    df = make_frame()
    df.loc[:4, "Total Fwd Packets"] = np.nan

    bundle = clean_frame(df)

    assert bundle.report["missing_values_imputed"] == 5
    assert not bundle.X.isna().any().any()


def test_missing_label_column_raises() -> None:
    df = make_frame().drop(columns=["Label"])
    with pytest.raises(KeyError):
        clean_frame(df)


def test_unknown_label_raises_when_verifying() -> None:
    df = make_frame(label="DDoS Hulk")
    with pytest.raises(KeyError):
        clean_frame(df, verify_labels=True)


def test_unknown_label_dropped_when_not_verifying() -> None:
    mixed = pd.concat(
        [make_frame(10, label="Benign"), make_frame(4, label="DDoS Hulk")],
        ignore_index=True,
    )
    bundle = clean_frame(mixed, verify_labels=False)

    assert len(bundle.X) == 10
    assert bundle.report["rows_dropped_unlabelled"] == 4


def test_metadata_columns_dropped() -> None:
    df = make_frame()
    for col in DROPPED_METADATA:
        df[col] = "x"

    bundle = clean_frame(df)

    assert not set(DROPPED_METADATA) & set(bundle.X.columns)


def test_apply_log1p_preserves_shape_and_row_order() -> None:
    df = pd.DataFrame({"Flow Bytes/s": [0.0, 1.0, 1e6, np.inf], "SYN Flag Count": [0, 1, 2, 3]})
    out = apply_log1p(df)

    assert out.shape == df.shape
    assert list(out.index) == list(df.index)
    # log1p(0) == 0 and log1p(1) == ln 2
    assert out["Flow Bytes/s"].iloc[0] == 0.0
    assert out["Flow Bytes/s"].iloc[1] == pytest.approx(np.log(2))


def test_apply_log1p_handles_negative_values() -> None:
    """Negative IAT deltas exist in CICIDS; log1p must not produce NaN."""
    df = pd.DataFrame({"Flow IAT Mean": [-100.0, -1.0, 0.0, 50.0]})
    out = apply_log1p(df)

    assert not np.isnan(out["Flow IAT Mean"].to_numpy()).any()
    assert out["Flow IAT Mean"].iloc[0] < 0  # sign preserved
