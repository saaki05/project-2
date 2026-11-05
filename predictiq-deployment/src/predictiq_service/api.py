"""FastAPI application: RUL prediction with conformal intervals, alerts and drift checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predictiq_service.bundle import INPUT_COLUMNS, RulBundle, load_bundle
from predictiq_service.drift import drift_report


class PredictRequest(BaseModel):
    rows: list[list[float]] = Field(
        description=f"Time-ordered readings, oldest first; columns: {', '.join(INPUT_COLUMNS)}")
    alert_threshold: float = Field(30.0, ge=0, description="RUL (cycles) that triggers a warning")


class DriftRequest(BaseModel):
    reference: list[list[float]]
    current: list[list[float]]


def create_app(bundle_path: str | Path | None = None, bundle: RulBundle | None = None) -> FastAPI:
    if bundle is None:
        if bundle_path is None:
            raise ValueError("provide bundle_path or bundle")
        bundle = load_bundle(bundle_path)
    app = FastAPI(title="PredictIQ", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "arch": bundle.arch, "trained_on": bundle.subset,
                "members": len(bundle.members), "window": bundle.window,
                "coverage_target": 1 - bundle.alpha, "n_input_columns": len(INPUT_COLUMNS)}

    @app.post("/predict")
    def predict(req: PredictRequest) -> dict:
        try:
            out = bundle.predict(np.asarray(req.rows))
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        # "critical" when even the optimistic end of the interval is below the threshold
        if out["upper"] <= req.alert_threshold:
            level = "critical"
        elif out["lower"] <= req.alert_threshold:
            level = "warning"
        else:
            level = "ok"
        return {**out, "alert": level, "alert_threshold": req.alert_threshold}

    @app.post("/drift")
    def drift(req: DriftRequest) -> dict:
        try:
            return drift_report(np.asarray(req.reference), np.asarray(req.current))
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err

    return app
