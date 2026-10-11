"""HTTP routes for detection, explanation and health."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from ml.registry import NIDSArtifact

from .config import settings
from .model_store import store
from .schemas import (
    BatchRequest,
    BatchResponse,
    ErrorResponse,
    Explanation,
    FlowFeatures,
    HealthResponse,
    MetricsResponse,
    PredictResponse,
    Verdict,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness plus whether a trained model is actually loaded."""
    return HealthResponse(
        status="ok" if store.loaded else "degraded",
        model_loaded=store.loaded,
        n_features=len(store.artifact.feature_names) if store.loaded else None,
        artifact_dir=str(settings.artifact_dir),
        version=settings.version,
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def predict(flow: FlowFeatures) -> PredictResponse:
    """Classify a single flow and return the hierarchical verdict."""
    try:
        verdict = store.predict_one(flow.features)
    except ValueError as exc:
        # A malformed feature dict is the caller's mistake, not a server fault.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PredictResponse(
        verdict=Verdict(**verdict),
        model_version=_model_version(),
    )


@router.post(
    "/predict/explain",
    response_model=PredictResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def predict_and_explain(flow: FlowFeatures) -> PredictResponse:
    """Classify a flow and explain why (global SHAP + local surrogate)."""
    try:
        result = store.explain(flow.features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PredictResponse(
        verdict=Verdict(**result["verdict"]),
        explanation=Explanation(
            verdict=Verdict(**result["verdict"]),
            shap=result.get("shap", []),
            surrogate_fidelity_r2=result.get("surrogate_fidelity_r2"),
            surrogate=result.get("surrogate") or None,
        ),
        model_version=_model_version(),
    )


@router.post(
    "/predict/batch",
    response_model=BatchResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def predict_batch(request: BatchRequest) -> BatchResponse:
    """Scan many flows at once and summarise the outcome."""
    try:
        verdicts = store.predict_many(request.rows)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return BatchResponse(
        total=len(verdicts),
        attacks=sum(1 for v in verdicts if v["is_attack"]),
        unseen=sum(1 for v in verdicts if v["unseen"]),
        verdicts=[Verdict(**v) for v in verdicts],
    )


@router.post("/predict/upload")
async def predict_upload(file: bytes) -> BatchResponse:
    """Score a CSV upload.

    Accepts raw bytes and parses in-process to keep the surface area small; the
    32 MB ceiling in settings bounds memory before parsing begins.
    """
    if len(file) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Upload of {len(file) / 1e6:.1f} MB exceeds the "
                   f"{settings.max_upload_bytes / 1e6:.0f} MB limit.",
        )

    try:
        import io

        frame = pd.read_csv(io.BytesIO(file))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    try:
        verdicts = store.require().predict(frame)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return BatchResponse(
        total=len(verdicts),
        attacks=sum(1 for v in verdicts if v["is_attack"]),
        unseen=sum(1 for v in verdicts if v["unseen"]),
        verdicts=[Verdict(**v) for v in verdicts],
    )


@router.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """Evaluation metrics captured during the training run."""
    path = Path(settings.artifact_dir) / "training_report.json"

    if not path.exists():
        return MetricsResponse(
            available=False,
            message="No training report found. Run `python -m ml.train`.",
        )

    try:
        return MetricsResponse(available=True, metrics=json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:  # noqa: BLE001
        return MetricsResponse(available=False, message=f"Unreadable report: {exc}")


@router.get("/features")
def features() -> dict:
    """The exact feature list the model expects, so clients can self-configure."""
    if not store.loaded:
        raise HTTPException(status_code=503, detail=store.load_error or "Model not loaded")
    artifact: NIDSArtifact = store.artifact
    return {
        "n_features": len(artifact.feature_names),
        "feature_names": artifact.feature_names,
        "metadata": artifact.metadata,
    }


def _model_version() -> str | None:
    if not store.loaded:
        return None
    return store.artifact.metadata.get("version", settings.version)
