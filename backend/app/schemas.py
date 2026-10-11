"""Request/response contracts shared by the API and the frontend.

``FlowFeatures`` uses ``extra="allow"`` on purpose: the CICIDS feature set is
wide and the selected subset may change as the pipeline evolves, so requiring
every column up front would break the client on every retrain.  Unknown fields
are ignored by :meth:`NIDSArtifact.align`, which picks only the features the
model was trained on.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FlowFeatures(BaseModel):
    """One or more network flows, keyed by CICIDS-2017 column name."""

    model_config = ConfigDict(extra="allow")

    features: dict[str, Any] = Field(
        ...,
        description="Raw CICIDS-2017 flow statistics, keyed by column name.",
        examples=[{"Total Fwd Packets": 12, "Flow Duration": 40000}],
    )


class BatchRequest(BaseModel):
    """A batch of flows submitted as a list of feature dicts."""

    model_config = ConfigDict(extra="forbid")

    rows: list[dict[str, Any]] = Field(
        ..., min_length=1, description="One dict of features per flow."
    )


class Verdict(BaseModel):
    """A hierarchical detection verdict for a single flow."""

    label: str
    label_id: int
    is_attack: bool
    unseen: bool
    level: int = Field(description="Which level produced the decision: 1, 2 or 3.")
    confidence: float
    p_attack: float
    anomaly_score: float


class Attribution(BaseModel):
    """One feature's contribution to a decision."""

    feature: str
    value: float
    contribution: float


class Explanation(BaseModel):
    """Local explanation bundle: SHAP plus a neighbour-based surrogate."""

    verdict: Verdict
    shap: list[Attribution]
    surrogate_fidelity_r2: float | None = None
    surrogate: list[Attribution] | None = None


class PredictResponse(BaseModel):
    """Response for a single-flow prediction, with optional explanation."""

    verdict: Verdict
    explanation: Explanation | None = None
    model_version: str | None = None


class BatchResponse(BaseModel):
    """Response for a multi-flow scan."""

    total: int
    attacks: int
    unseen: int
    verdicts: list[Verdict]


class HealthResponse(BaseModel):
    """Liveness plus whether a trained model is actually loaded."""

    status: str
    model_loaded: bool
    n_features: int | None = None
    artifact_dir: str
    version: str


class MetricsResponse(BaseModel):
    """Evaluation metrics captured during the training run."""

    available: bool
    metrics: dict[str, Any] | None = None
    message: str | None = None


class ErrorResponse(BaseModel):
    """Uniform error shape so the frontend can render failures consistently."""

    detail: str
    code: str = "bad_request"
