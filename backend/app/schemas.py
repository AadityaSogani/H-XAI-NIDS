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


# ---------------------------------------------------------------------------
# Prevention (IPS) contracts.
#
# These sit *beside* the detection contracts, never replacing them: detection
# still returns a ``Verdict``; prevention wraps that verdict in a countermeasure
# decision.  ``FlowContext`` is deliberately separate from ``FlowFeatures`` — it
# carries IP/port metadata the detector never sees (those columns are dropped at
# training time) and is used only to key a blocklist entry.
# ---------------------------------------------------------------------------


class FlowContext(BaseModel):
    """Optional, model-irrelevant metadata used to key a prevention action.

    None of these fields are fed to the detector — they exist so a prevention
    decision has something concrete to act on (an IP to block, a port to drop).
    """

    model_config = ConfigDict(extra="forbid")

    src_ip: str | None = Field(
        default=None, description="Source IP to block/rate-limit, if known."
    )
    dst_ip: str | None = Field(default=None, description="Destination IP, if known.")
    dst_port: int | None = Field(
        default=None, ge=0, le=65535, description="Destination port, if known."
    )
    protocol: str | None = Field(
        default=None, description="L4 protocol hint, e.g. 'tcp' / 'udp'."
    )


class PreventionRequest(BaseModel):
    """A flow to detect-and-prevent: features + optional context + mode."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, Any] = Field(
        ...,
        description="Raw CICIDS-2017 flow statistics, keyed by column name.",
    )
    context: FlowContext | None = Field(
        default=None, description="Optional metadata used only to key the action."
    )
    mode: str = Field(
        default="prevent",
        description="'prevent' enforces; 'monitor' is a dry run that never blocks.",
        pattern="^(prevent|monitor)$",
    )


class PreventionAction(BaseModel):
    """The countermeasure chosen for a verdict, with its provenance."""

    action: str = Field(description="ALLOW, ALERT, RATE_LIMIT, DROP, BLOCK or QUARANTINE.")
    enforced: bool = Field(description="True if this call actually added a blocklist entry.")
    gated: bool = Field(
        description="True if an attack was downgraded to ALERT by the confidence gate."
    )
    escalated: bool = Field(description="True if this subject was already blocked (repeat offender).")
    severity: int = Field(description="0 (info) .. 5 (critical); matches the UI severity ramp.")
    ttl_seconds: int = Field(description="How long the enforced block lasts; 0 if none.")
    rationale: str = Field(description="Human-readable reason for the decision.")
    rule: str = Field(description="A representative packet-filter rule an operator could apply.")
    subject: str | None = Field(description="Blocklist key acted on, e.g. 'ip:1.2.3.4'.")


class PreventionDecision(BaseModel):
    """A detection verdict bundled with its prevention action."""

    verdict: Verdict
    prevention: PreventionAction
    model_version: str | None = None


class BlockEntry(BaseModel):
    """One active entry on the prevention blocklist."""

    key: str
    action: str
    severity: int
    reason: str
    hits: int
    created_at: float
    expires_at: float
    ttl_remaining: float


class BlocklistResponse(BaseModel):
    """The live blocklist, soonest-expiry first."""

    total: int
    entries: list[BlockEntry]

