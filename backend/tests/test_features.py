"""Unit tests for Stage-2 domain features and Stage-3 SFE embedding."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features import DOMAIN_FEATURES, add_domain_features, describe_domain_features
from ml.sfe import N_SFE_FEATURES, SFETransformer


@pytest.fixture
def flow_frame() -> pd.DataFrame:
    """A small frame with every column the domain features reference."""
    rng = np.random.default_rng(7)
    n = 200
    return pd.DataFrame(
        {
            "Fwd Packets Length Total": rng.integers(100, 5000, n).astype(float),
            "Bwd Packets Length Total": rng.integers(100, 5000, n).astype(float),
            "Total Fwd Packets": rng.integers(1, 50, n).astype(float),
            "Total Backward Packets": rng.integers(1, 50, n).astype(float),
            "Flow IAT Mean": rng.uniform(10, 1000, n),
            "Flow IAT Std": rng.uniform(1, 500, n),
            "Fwd IAT Total": rng.uniform(1000, 100_000, n),
            "Flow IAT Min": rng.uniform(1, 100, n),
            "Packet Length Mean": rng.uniform(50, 800, n),
            "Packet Length Std": rng.uniform(5, 200, n),
            "Fwd Packets/s": rng.uniform(0, 50, n),
            "Bwd Packets/s": rng.uniform(0, 50, n),
            "Packet Length Max": rng.uniform(800, 1500, n),
            "Packet Length Min": rng.uniform(0, 50, n),
        }
    )


# ---------------------------------------------------------------- domain ----


def test_domain_features_are_appended(flow_frame):
    out, new_cols = add_domain_features(flow_frame)

    assert new_cols == DOMAIN_FEATURES
    assert all(c in out.columns for c in DOMAIN_FEATURES)
    assert out.shape == (len(flow_frame), flow_frame.shape[1] + len(DOMAIN_FEATURES))


def test_domain_features_have_no_nan_or_inf(flow_frame):
    out, new_cols = add_domain_features(flow_frame)
    block = out[new_cols]

    assert not block.isna().any().any()
    assert not np.isinf(block.to_numpy()).any()


def test_zero_byte_flow_does_not_divide_by_zero(flow_frame):
    """A flow with no bytes either way is valid; must yield 0, not NaN."""
    frame = flow_frame.copy()
    frame["Fwd Packets Length Total"] = 0.0
    frame["Bwd Packets Length Total"] = 0.0

    out, new_cols = add_domain_features(frame)

    assert not out[new_cols].isna().any().any()
    assert out["Payload_Ratio_Fwd"].iloc[0] == 0.0
    assert out["Byte_Entropy"].iloc[0] == 0.0


def test_fully_forward_flow_saturates_ratio(flow_frame):
    frame = flow_frame.copy()
    frame["Bwd Packets Length Total"] = 0.0

    out, _ = add_domain_features(frame)

    assert np.allclose(out["Payload_Ratio_Fwd"], 1.0)
    assert np.allclose(out["Byte_Asymmetry_Index"], 1.0)
    assert np.allclose(out["Byte_Entropy"], 0.0)


def test_balanced_flow_has_zero_asymmetry(flow_frame):
    frame = flow_frame.copy()
    frame["Bwd Packets Length Total"] = frame["Fwd Packets Length Total"]

    out, _ = add_domain_features(frame)

    assert np.allclose(out["Payload_Ratio_Fwd"], 0.5)
    assert np.allclose(out["Byte_Asymmetry_Index"], 0.0)


def test_entropy_peaks_for_balanced_flow(flow_frame):
    """Entropy is maximal (1 bit) at a 50/50 byte split."""
    frame = flow_frame.copy()
    frame["Bwd Packets Length Total"] = frame["Fwd Packets Length Total"]

    out, _ = add_domain_features(frame)

    assert np.allclose(out["Byte_Entropy"], 1.0)


def test_missing_columns_are_tolerated():
    """Domain features must still build on a reduced feature set."""
    frame = pd.DataFrame({"Total Fwd Packets": [1.0, 2.0, 3.0]})
    out, new_cols = add_domain_features(frame)

    assert all(c in out.columns for c in new_cols)
    assert not out[new_cols].isna().any().any()
    assert not np.isinf(out[new_cols].to_numpy()).any()


def test_empty_frame_returns_unchanged():
    empty = pd.DataFrame({"Total Fwd Packets": pd.Series([], dtype=float)})
    out, new_cols = add_domain_features(empty)

    assert new_cols == []
    assert out.empty


def test_descriptions_cover_every_feature():
    described = {d["name"] for d in describe_domain_features()}
    assert described == set(DOMAIN_FEATURES)


# -------------------------------------------------------------------- SFE ----


def test_sfe_produces_nine_features(flow_frame):
    sfe = SFETransformer(n_clusters=5, n_components=4)
    out = sfe.fit_transform(flow_frame)

    assert out.shape == (len(flow_frame), N_SFE_FEATURES)
    assert len(sfe.feature_names_) == N_SFE_FEATURES


def test_sfe_output_is_finite(flow_frame):
    out = SFETransformer().fit_transform(flow_frame)

    assert not np.isnan(out.to_numpy()).any()
    assert not np.isinf(out.to_numpy()).any()


def test_sfe_transform_matches_fit_transform(flow_frame):
    """transform() must be a pure function of the fitted state."""
    sfe = SFETransformer().fit(flow_frame)

    assert np.allclose(
        sfe.transform(flow_frame).to_numpy(),
        sfe.fit_transform(flow_frame).to_numpy(),
    )


def test_sfe_transform_requires_fit(flow_frame):
    with pytest.raises(RuntimeError):
        SFETransformer().transform(flow_frame)


def test_sfe_handles_unseen_rows(flow_frame):
    """Serving must cope with a single flow (batch size 1)."""
    sfe = SFETransformer().fit(flow_frame)
    out = sfe.transform(flow_frame.iloc[[0]])

    assert out.shape == (1, N_SFE_FEATURES)


def test_sfe_is_deterministic(flow_frame):
    a = SFETransformer().fit_transform(flow_frame)
    b = SFETransformer().fit_transform(flow_frame)

    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_sfe_clamps_too_many_clusters():
    """k must not exceed the row count, or KMeans raises."""
    tiny = pd.DataFrame(np.random.default_rng(0).normal(size=(4, 3)))
    out = SFETransformer(n_clusters=50, n_components=10).fit_transform(tiny)

    assert out.shape[0] == 4
    assert out.shape[1] <= N_SFE_FEATURES


def test_sfe_returns_a_fresh_range_index(flow_frame):
    """The embedding must NOT carry the input index forward.

    A corpus sliced with ``.iloc[train_idx]`` has non-contiguous, possibly
    duplicated labels.  Preserving them made ``pd.concat(axis=1)`` outer-join on
    the labels and silently duplicate rows (315,972 rows became 410,780).
    """
    indexed = flow_frame.copy()
    indexed.index = [f"row{i}" for i in range(len(indexed))]

    out = SFETransformer().fit_transform(indexed)

    assert list(out.index) == list(range(len(indexed)))


def test_sfe_concat_with_reset_frame_does_not_duplicate_rows(flow_frame):
    """Regression test for the index-join row blow-up."""
    frame = flow_frame.copy()
    frame.index = np.repeat(np.arange(len(frame)), 2)[: len(frame)]  # dup labels

    emb = SFETransformer().fit_transform(frame)
    joined = pd.concat([frame.reset_index(drop=True), emb], axis=1)

    assert len(joined) == len(frame)
    assert not joined.isna().any().any()

    described = {d["name"] for d in describe_domain_features()}
    assert described == set(DOMAIN_FEATURES)
