"""Model persistence and the serving-time inference bundle.

Training and serving are separated deliberately.  Training needs the full
corpus and 16 GB; the API needs a fast cold start and a small memory
footprint.  This module is the contract between them: it serialises exactly the
fitted state required to reproduce a prediction, and nothing else.

The critical detail is the *feature contract*.  A prediction is only valid if
the serving-time row is built from the same columns, in the same order, as the
row the models were trained on.  We therefore persist the feature list and
reorder (or reject) incoming frames against it, rather than trusting the
caller's column order.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MANIFEST = "manifest.json"


@dataclass(slots=True)
class NIDSArtifact:
    """Everything needed to score a flow without retraining."""

    model: object                       # HierarchicalDetector
    feature_names: list[str]
    scaler_state: dict | None = None    # fitted separately, if used
    metadata: dict = field(default_factory=dict)

    def save(self, directory: Path | str) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, directory / "model.joblib", compress=3)
        (directory / MANIFEST).write_text(
            json.dumps(
                {
                    "feature_names": self.feature_names,
                    "scaler_state": self.scaler_state,
                    "metadata": self.metadata,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, directory: Path | str) -> "NIDSArtifact":
        directory = Path(directory)
        manifest_path = directory / MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(f"No {MANIFEST} in {directory}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return cls(
            model=joblib.load(directory / "model.joblib"),
            feature_names=manifest["feature_names"],
            scaler_state=manifest.get("scaler_state"),
            metadata=manifest.get("metadata", {}),
        )

    # ------------------------------------------------------------ scoring ----

    def align(self, frame: pd.DataFrame) -> np.ndarray:
        """Reorder/validate a frame into the exact training column order.

        Raises
        ------
        ValueError
            If a required feature is absent.  Silently zero-filling a missing
            column would produce a confident-looking verdict computed from the
            wrong numbers, which is worse than refusing the request.
        """
        missing = [c for c in self.feature_names if c not in frame.columns]
        if missing:
            raise ValueError(
                f"Missing {len(missing)} required feature(s): {missing[:5]}"
                + (" ..." if len(missing) > 5 else "")
            )

        aligned = frame[self.feature_names]
        values = aligned.to_numpy(dtype=np.float64)

        if np.isnan(values).any():
            # Impute per-column with the median so one bad sensor reading does
            # not poison an otherwise valid flow.
            col_median = np.nanmedian(
                np.where(np.isnan(values), np.nan, values), axis=0
            )
            col_median = np.where(np.isnan(col_median), 0.0, col_median)
            values = np.where(np.isnan(values), col_median, values)

        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return values

    def predict(self, frame: pd.DataFrame) -> list[dict]:
        """Score one or more flows, returning hierarchical verdicts."""
        return self.model.predict(self.align(frame))

    def predict_proba_attack(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba_attack(self.align(frame))
