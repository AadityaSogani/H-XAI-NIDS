"""Lifetime management for the trained model.

The artifact is loaded **once** during FastAPI's lifespan startup, not per
request.  Deserialising a random forest with 200 trees on every call would
dominate latency and defeat the point of serving.

If no artifact exists the app still starts — it just reports
``model_loaded: false`` and refuses predictions.  Booting successfully with a
broken model is better than crashing the container, because ``/health`` then
tells an operator exactly what is wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.explain import SurrogateExplainer, TreeShapExplainer
from ml.registry import NIDSArtifact

from .config import settings


class ModelStore:
    """Holds the loaded artifact plus its lazily-built explainers."""

    def __init__(self) -> None:
        self.artifact: NIDSArtifact | None = None
        self._background: np.ndarray | None = None
        self._shap: TreeShapExplainer | None = None
        self._surrogate: SurrogateExplainer | None = None
        self.load_error: str | None = None

    # ------------------------------------------------------------ lifecycle ----

    def load(self) -> bool:
        """Attempt to load the artifact.  Returns True on success."""
        try:
            self.artifact = NIDSArtifact.load(settings.artifact_path)
            self.load_error = None
            self._load_background()
            return True
        except FileNotFoundError as exc:
            self.artifact = None
            self.load_error = str(exc)
            return False
        except Exception as exc:  # noqa: BLE001
            self.artifact = None
            self.load_error = f"{type(exc).__name__}: {exc}"
            return False

    def _load_background(self) -> None:
        """Attach the reference distribution used for explanations.

        Optional: without it SHAP and the surrogate fall back to a single zero
        row, so explanations still render (just with less context).
        """
        path = settings.artifact_path / "background.npy"
        if path.exists():
            try:
                self.set_background(np.load(path))
            except Exception:  # noqa: BLE001
                self._background = None

    @property
    def loaded(self) -> bool:
        return self.artifact is not None

    def require(self) -> NIDSArtifact:
        """Return the artifact or raise a clear error for the 503 handler."""
        if self.artifact is None:
            raise RuntimeError(
                self.load_error
                or "No trained model available. Run `python -m ml.train` first."
            )
        return self.artifact

    # ------------------------------------------------------------ predict -----

    def predict_one(self, features: dict) -> dict:
        frame = pd.DataFrame([features])
        return self.require().predict(frame)[0]

    def predict_many(self, rows: list[dict]) -> list[dict]:
        if len(rows) > settings.max_batch_rows:
            raise ValueError(
                f"Batch of {len(rows)} exceeds the limit of "
                f"{settings.max_batch_rows} rows."
            )
        frame = pd.DataFrame(rows)
        return self.require().predict(frame)

    # ------------------------------------------------------------ explain -----

    def explain(self, features: dict) -> dict:
        """Verdict plus global SHAP and a local surrogate explanation.

        The surrogate's fidelity is reported rather than hidden: a low R^2 means
        the local explanation should not be trusted, and an analyst deserves to
        know that.
        """
        artifact = self.require()
        frame = pd.DataFrame([features])
        aligned = artifact.align(frame)

        verdict = artifact.predict(frame)[0]

        result: dict = {"verdict": verdict}

        try:
            shap_ranking = self._shap_explainer().local_explanation(
                aligned[0], top_k=10
            )
            result["shap"] = shap_ranking
        except Exception:  # noqa: BLE001
            # Explanations are advisory; a SHAP failure must not lose the alert.
            result["shap"] = []

        try:
            surrogate = self._surrogate_explainer().explain(
                aligned[0], self._background_rows()
            )
            result["surrogate_fidelity_r2"] = surrogate["fidelity_r2"]
            result["surrogate"] = surrogate["contributions"][:10]
        except Exception:  # noqa: BLE001
            result["surrogate_fidelity_r2"] = None
            result["surrogate"] = []

        return result

    def _shap_explainer(self) -> TreeShapExplainer:
        if self._shap is None:
            artifact = self.require()
            self._shap = TreeShapExplainer(
                artifact.model.level1_,
                artifact.feature_names,
                background=self._background_rows(),
            )
        return self._shap

    def _surrogate_explainer(self) -> SurrogateExplainer:
        if self._surrogate is None:
            artifact = self.require()
            self._surrogate = SurrogateExplainer(
                lambda Z: artifact.model.predict_proba_attack(Z),
                artifact.feature_names,
            )
        return self._surrogate

    def _background_rows(self) -> np.ndarray:
        """Rows used as the SHAP/surrogate reference distribution.

        Falls back to the explained row's own zeros when no background was
        captured, so explanations still render rather than erroring.
        """
        if self._background is None:
            artifact = self.require()
            self._background = np.zeros(
                (1, len(artifact.feature_names)), dtype=np.float64
            )
        return self._background

    def set_background(self, rows: np.ndarray) -> None:
        self._background = np.asarray(rows, dtype=np.float64)


store = ModelStore()
