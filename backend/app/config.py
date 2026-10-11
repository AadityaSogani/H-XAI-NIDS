"""Runtime configuration, read once from the environment.

Every value has a working default so the API starts with no configuration at
all.  ``ARTIFACT_DIR`` is the only path that matters: the API refuses to serve
predictions from an untrained model rather than returning random verdicts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable settings resolved at import time."""

    artifact_dir: Path = BACKEND_ROOT / "artifacts"
    cors_origins: tuple[str, ...] = (
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    )
    max_batch_rows: int = 5_000
    max_upload_bytes: int = 32 * 1024 * 1024  # 32 MB
    app_name: str = "H-XAI-NIDS API"
    version: str = "1.0.0"

    @classmethod
    def from_env(cls) -> "Settings":
        origins = os.getenv("CORS_ORIGINS", "")
        default_origins = (
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        )
        return cls(
            artifact_dir=Path(
                os.getenv("ARTIFACT_DIR", str(BACKEND_ROOT / "artifacts"))
            ),
            cors_origins=tuple(
                o.strip() for o in origins.split(",") if o.strip()
            )
            or default_origins,
            max_batch_rows=int(os.getenv("MAX_BATCH_ROWS", "5000")),
        )

    @property
    def artifact_path(self) -> Path:
        return self.artifact_dir


settings = Settings.from_env()
