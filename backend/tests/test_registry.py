"""Unit tests for artifact persistence and feature alignment."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.hierarchy import HierarchicalDetector
from ml.registry import NIDSArtifact


@pytest.fixture
def tiny_model():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(200, 3))
    y = np.concatenate([np.zeros(150), np.ones(50)]).astype(int)
    detector = HierarchicalDetector(random_state=0).fit(X, y)
    return detector, X, y


@pytest.fixture
def frame():
    rng = np.random.default_rng(8)
    return pd.DataFrame(rng.normal(size=(5, 3)), columns=["a", "b", "c"])


def test_roundtrip_preserves_predictions(tiny_model, frame, tmp_path):
    detector, _, _ = tiny_model
    artifact = NIDSArtifact(
        model=detector,
        feature_names=["a", "b", "c"],
        metadata={"trained_on": "test"},
    )

    before = artifact.predict(frame)
    reloaded = NIDSArtifact.load(artifact.save(tmp_path))
    after = reloaded.predict(frame)

    assert before == after
    assert reloaded.metadata["trained_on"] == "test"


def test_save_creates_expected_files(tiny_model, frame, tmp_path):
    detector, _, _ = tiny_model
    directory = NIDSArtifact(detector, ["a", "b", "c"]).save(tmp_path)

    assert (directory / "model.joblib").exists()
    assert (directory / "manifest.json").exists()


def test_load_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        NIDSArtifact.load(tmp_path / "does_not_exist")


def test_align_reorders_shuffled_columns(tiny_model, frame):
    detector, _, _ = tiny_model
    artifact = NIDSArtifact(detector, ["a", "b", "c"])

    shuffled = frame[["c", "a", "b"]]
    aligned = artifact.align(shuffled)

    assert np.allclose(aligned, frame[["a", "b", "c"]].to_numpy())


def test_align_rejects_missing_features(tiny_model, frame):
    detector, _, _ = tiny_model
    artifact = NIDSArtifact(detector, ["a", "b", "c", "d"])

    with pytest.raises(ValueError, match="Missing"):
        artifact.align(frame)


def test_align_imputes_nan_with_the_column_median(tiny_model, frame):
    detector, _, _ = tiny_model
    artifact = NIDSArtifact(detector, ["a", "b", "c"])

    frame = frame.copy()
    frame.loc[1, "a"] = np.nan
    aligned = artifact.align(frame)

    assert not np.isnan(aligned).any()
    others = frame["a"].dropna().to_numpy()
    assert aligned[1, 0] == pytest.approx(np.median(others))


def test_align_neutralises_infinities(tiny_model, frame):
    detector, _, _ = tiny_model
    artifact = NIDSArtifact(detector, ["a", "b", "c"])

    frame = frame.copy()
    frame.loc[0, "b"] = np.inf
    frame.loc[2, "b"] = -np.inf

    assert np.isfinite(artifact.align(frame)).all()


def test_predict_matches_direct_model_call(tiny_model, frame):
    detector, _, _ = tiny_model
    artifact = NIDSArtifact(detector, ["a", "b", "c"])

    assert artifact.predict(frame) == detector.predict(
        frame.to_numpy(dtype=np.float64)
    )


def test_predict_proba_attack_shape(tiny_model, frame):
    detector, _, _ = tiny_model
    artifact = NIDSArtifact(detector, ["a", "b", "c"])

    proba = artifact.predict_proba_attack(frame)

    assert proba.shape == (len(frame),)
    assert ((proba >= 0) & (proba <= 1)).all()
