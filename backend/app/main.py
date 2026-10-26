"""FastAPI application entry point.

Run from the ``backend`` directory::

    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .model_store import store
from .routes import router
from .prevention_routes import router as prevention_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained artifact once, at startup rather than per request."""
    loaded = store.load()
    if loaded:
        print(f"[startup] model loaded from {settings.artifact_dir}")
    else:
        print(f"[startup] WARNING: no model loaded - {store.load_error}")
        print("[startup] run `python -m ml.train` to produce artifacts/")
    yield
    print("[shutdown] releasing model")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "Hierarchical Contextual Stacking Network Intrusion Detection System. "
        "Level 1 detects attacks, Level 2 classifies the family, Level 3 flags "
        "potential zero-day traffic, and every alert can be explained."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(prevention_router)


@app.get("/", tags=["meta"])
def root() -> dict:
    """Service banner with pointers to the useful endpoints."""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "model_loaded": store.loaded,
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict",
            "predict_explained": "POST /predict/explain",
            "predict_batch": "POST /predict/batch",
            "predict_upload": "POST /predict/upload",
            "prevent": "POST /prevent",
            "blocklist": "GET /prevent/blocklist",
            "metrics": "/metrics",
            "features": "/features",
            "docs": "/docs",
        },
    }
