"""HTTP routes for the intrusion *prevention* layer (IPS).

These routes sit beside ``routes.py`` and never replace it: detection still
lives at ``/predict*`` and is untouched.  Prevention reuses ``store.predict_one``
— there is exactly one detector — and wraps its verdict in a countermeasure via
:class:`ml.prevention.PreventionEngine`.

Enforcement is simulated.  ``/prevent`` records a decision and, in ``prevent``
mode, maintains a live TTL blocklist the UI acts on; it does not drop packets on
the wire.  A real deployment would wire ``PreventionAction.rule`` into the host
firewall.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ml.prevention import PreventionEngine

from .config import settings
from .model_store import store
from .schemas import (
    BlockEntry,
    BlocklistResponse,
    ErrorResponse,
    PreventionAction,
    PreventionDecision,
    PreventionRequest,
    Verdict,
)

router = APIRouter(tags=["prevention"])

# One engine for the process: the blocklist is shared state, exactly as the
# model store is.  A fresh engine per request would forget every prior block.
engine = PreventionEngine()


@router.post(
    "/prevent",
    response_model=PreventionDecision,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def prevent(request: PreventionRequest) -> PreventionDecision:
    """Detect a flow and return (and optionally enforce) a countermeasure.

    ``mode="prevent"`` adds an enforcing action to the live blocklist;
    ``mode="monitor"`` is a dry run that reports the would-be action without
    disrupting anything — useful for tuning the policy before it bites.
    """
    try:
        verdict = store.predict_one(request.features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    context = request.context.model_dump() if request.context else None
    prevention = engine.decide(verdict, context, mode=request.mode)

    return PreventionDecision(
        verdict=Verdict(**verdict),
        prevention=PreventionAction(**prevention),
        model_version=_model_version(),
    )


@router.get("/prevent/blocklist", response_model=BlocklistResponse)
def blocklist() -> BlocklistResponse:
    """The live blocklist: every subject currently blocked, soonest-expiry first.

    Expired entries are dropped on read, so what is listed is exactly what is
    currently enforced.
    """
    entries = engine.blocklist.entries()
    return BlocklistResponse(
        total=len(entries),
        entries=[BlockEntry(**e) for e in entries],
    )


@router.delete(
    "/prevent/blocklist/{key}",
    responses={404: {"model": ErrorResponse}},
)
def unblock(key: str) -> dict:
    """Manually clear one block before its TTL expires (analyst override)."""
    if not engine.unblock(key):
        raise HTTPException(
            status_code=404,
            detail=f"No active block for key {key!r} (it may have expired).",
        )
    return {"removed": key, "remaining": len(engine.blocklist.entries())}


def _model_version() -> str | None:
    if not store.loaded:
        return None
    return store.artifact.metadata.get("version", settings.version)

