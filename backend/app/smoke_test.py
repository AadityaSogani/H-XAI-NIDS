"""Live smoke test for the FastAPI backend.

Exercises every endpoint with TestClient (no separate server needed), using a
real flow from the trained artifact's background plus a deliberately malformed
request to confirm error handling.

Run from the ``backend`` directory::

    python -m app.smoke_test
"""
from __future__ import annotations

from pathlib import Path
import sys

# Make `backend/` importable so `ml.*` and `app.*` both resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from ml.registry import NIDSArtifact  # noqa: E402


def main() -> int:
    if not (settings.artifact_path / "manifest.json").exists():
        print("No trained artifact. Run `python -m ml.train` first.")
        return 1

    artifact = NIDSArtifact.load(settings.artifact_path)
    background = np.load(settings.artifact_path / "background.npy")
    features = artifact.feature_names

    print(f"artifact: {len(features)} features, background {background.shape}")

    with TestClient(app) as client:
        failures = 0

        # --- health ---------------------------------------------------------
        r = client.get("/health")
        health = r.json()
        print(f"\nGET  /health        -> {r.status_code}  model_loaded={health['model_loaded']} "
              f"n_features={health['n_features']}")
        failures += r.status_code != 200 or not health["model_loaded"]

        # --- root -----------------------------------------------------------
        r = client.get("/")
        print(f"GET  /              -> {r.status_code}  {r.json()['name']}")
        failures += r.status_code != 200

        # --- features -------------------------------------------------------
        r = client.get("/features")
        print(f"GET  /features      -> {r.status_code}  {r.json()['n_features']} features")
        failures += r.status_code != 200

        # --- metrics --------------------------------------------------------
        r = client.get("/metrics")
        payload = r.json()
        pr_auc = payload["metrics"]["level1"]["pr_auc"] if payload["available"] else None
        print(f"GET  /metrics       -> {r.status_code}  available={payload['available']} "
              f"PR-AUC={pr_auc}")
        failures += r.status_code != 200

        # --- single prediction (real flow from the background) --------------
        row = {name: float(background[0][i]) for i, name in enumerate(features)}
        r = client.post("/predict", json={"features": row})
        verdict = r.json()["verdict"]
        print(f"POST /predict       -> {r.status_code}  {verdict['label']} "
              f"(level {verdict['level']}, conf {verdict['confidence']})")
        failures += r.status_code != 200

        # --- explained prediction ------------------------------------------
        r = client.post("/predict/explain", json={"features": row})
        body = r.json()
        expl = body.get("explanation") or {}
        n_shap = len(expl.get("shap", []))
        fidelity = expl.get("surrogate_fidelity_r2")
        print(f"POST /predict/explain -> {r.status_code}  {body['verdict']['label']}, "
              f"{n_shap} SHAP terms, surrogate R2={fidelity}")
        if n_shap:
            top = expl["shap"][0]
            print(f"     top driver: {top['feature']} = {top['value']} "
                  f"(contribution {top['contribution']})")
        failures += r.status_code != 200

        # --- batch ----------------------------------------------------------
        batch_rows = [
            {name: float(background[i][j]) for j, name in enumerate(features)}
            for i in range(25)
        ]
        r = client.post("/predict/batch", json={"rows": batch_rows})
        summary = r.json()
        print(f"POST /predict/batch -> {r.status_code}  total={summary['total']} "
              f"attacks={summary['attacks']} unseen={summary['unseen']}")
        failures += r.status_code != 200

        # --- error handling -------------------------------------------------
        r = client.post("/predict", json={"features": {"Total Fwd Packets": 1}})
        print(f"POST /predict (missing features) -> {r.status_code}  "
              f"{r.json()['detail'][:60]}...")
        failures += r.status_code != 400

    print(f"\n{'ALL SMOKE TESTS PASSED' if failures == 0 else f'{failures} FAILURES'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
