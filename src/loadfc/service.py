"""FastAPI inference service.

Endpoints:

* ``POST /forecast`` -- hand it recent observed load, get the next hours back.
* ``GET  /health``   -- liveness plus what model is loaded.
* ``GET  /drift``    -- PSI of the most recent requests against training data.
* ``GET  /metrics``  -- Prometheus exposition format.

Two decisions worth stating. First, the service never silently extrapolates:
a request whose history is too short, has gaps, or asks for more hours than
the model's horizon is rejected with a 422 explaining which rule it broke.
A forecasting service that guesses when it is under-informed is worse than one
that refuses, because nobody downstream can tell the difference.

Second, every served request is pushed into a ring buffer that the drift
endpoint scores against the training distribution. Monitoring that needs a
separate batch job tends not to get run.
"""

from __future__ import annotations

import os
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from loadfc.drift import DriftMonitor
from loadfc.features import LeakageError, build_inference_features
from loadfc.model import ModelBundle

MODEL_PATH = Path(os.environ.get("LOADFC_MODEL", "models/model.joblib"))
REFERENCE_PATH = Path(os.environ.get("LOADFC_REFERENCE", "models/reference.parquet"))
RING_SIZE = int(os.environ.get("LOADFC_RING_SIZE", "2000"))


class Observation(BaseModel):
    """One historical hourly reading."""

    timestamp: str
    load_mw: float = Field(gt=0, description="metered load in megawatts")


class ForecastRequest(BaseModel):
    history: list[Observation] = Field(min_length=1)
    steps: int = Field(default=24, ge=1, le=24)

    @field_validator("history")
    @classmethod
    def _reject_empty(cls, value: list[Observation]) -> list[Observation]:
        if not value:
            raise ValueError("history must not be empty")
        return value


class ForecastPoint(BaseModel):
    timestamp: str
    load_mw: float


class ForecastResponse(BaseModel):
    forecast: list[ForecastPoint]
    model_version: str
    horizon_hours: int
    latency_ms: float


class _State:
    """Process-wide handles, populated on startup."""

    bundle: ModelBundle | None = None
    monitor: DriftMonitor | None = None
    recent: deque[pd.DataFrame] = deque(maxlen=RING_SIZE)
    counters: dict[str, float] = {
        "requests_total": 0.0,
        "errors_total": 0.0,
        "forecast_hours_total": 0.0,
        "latency_ms_sum": 0.0,
    }


state = _State()


def load_artifacts(
    model_path: Path = MODEL_PATH, reference_path: Path = REFERENCE_PATH
) -> None:
    """Load the model, and the drift reference if one was exported with it."""
    if model_path.exists():
        state.bundle = ModelBundle.load(model_path)
    if reference_path.exists():
        state.monitor = DriftMonitor(pd.read_parquet(reference_path))


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Load forecasting service",
    version="0.1.0",
    description="Day-ahead electricity load forecasting for the PJM East zone.",
    lifespan=lifespan,
)


def _to_series(history: list[Observation]) -> pd.Series:
    """Validate and convert the posted history into an hourly series."""
    frame = pd.DataFrame({"timestamp": [o.timestamp for o in history],
                          "load_mw": [o.load_mw for o in history]})
    try:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    except (ValueError, TypeError) as error:
        raise HTTPException(422, f"could not parse timestamps: {error}") from error

    series = frame.set_index("timestamp")["load_mw"].sort_index()
    if series.index.has_duplicates:
        raise HTTPException(422, "history contains duplicate timestamps")

    gaps = series.index.to_series().diff().dropna()
    if not gaps.empty and (gaps != pd.Timedelta(hours=1)).any():
        bad = gaps[gaps != pd.Timedelta(hours=1)]
        raise HTTPException(
            422,
            f"history must be contiguous hourly readings; found {len(bad)} gap(s), "
            f"first at {bad.index[0]}",
        )
    return series


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    """Forecast the hours immediately following the supplied history."""
    if state.bundle is None:
        raise HTTPException(503, f"no model loaded; expected one at {MODEL_PATH}")

    started = time.perf_counter()
    state.counters["requests_total"] += 1

    series = _to_series(request.history)
    try:
        features = build_inference_features(series, state.bundle.spec, request.steps)
    except (LeakageError, ValueError) as error:
        state.counters["errors_total"] += 1
        raise HTTPException(422, str(error)) from error

    predictions = state.bundle.predict(features)
    state.recent.append(features)

    latency_ms = (time.perf_counter() - started) * 1e3
    state.counters["forecast_hours_total"] += len(predictions)
    state.counters["latency_ms_sum"] += latency_ms

    return ForecastResponse(
        forecast=[
            ForecastPoint(timestamp=str(ts), load_mw=round(float(value), 1))
            for ts, value in zip(features.index, predictions, strict=True)
        ],
        model_version=state.bundle.trained_on.get("end", "unknown"),
        horizon_hours=state.bundle.spec.horizon,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness, and enough detail to tell which artefact is serving."""
    return {
        "status": "ok" if state.bundle is not None else "degraded",
        "model_loaded": state.bundle is not None,
        "drift_reference_loaded": state.monitor is not None,
        "trained_on": state.bundle.trained_on if state.bundle else None,
        "horizon_hours": state.bundle.spec.horizon if state.bundle else None,
    }


@app.get("/drift")
def drift() -> dict[str, Any]:
    """Score recently served requests against the training distribution."""
    if state.monitor is None:
        raise HTTPException(503, f"no drift reference loaded; expected {REFERENCE_PATH}")
    if not state.recent:
        return {"status": "no_data", "n_live": 0}

    live = pd.concat(list(state.recent))
    return state.monitor.score(live).to_dict()


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    """Prometheus exposition format, hand-rolled to avoid a client dependency."""
    requests = state.counters["requests_total"]
    mean_latency = state.counters["latency_ms_sum"] / requests if requests else 0.0

    lines = [
        "# HELP loadfc_requests_total Forecast requests received.",
        "# TYPE loadfc_requests_total counter",
        f"loadfc_requests_total {requests:.0f}",
        "# HELP loadfc_errors_total Requests rejected as unservable.",
        "# TYPE loadfc_errors_total counter",
        f"loadfc_errors_total {state.counters['errors_total']:.0f}",
        "# HELP loadfc_forecast_hours_total Forecast hours produced.",
        "# TYPE loadfc_forecast_hours_total counter",
        f"loadfc_forecast_hours_total {state.counters['forecast_hours_total']:.0f}",
        "# HELP loadfc_latency_ms_mean Mean end-to-end forecast latency.",
        "# TYPE loadfc_latency_ms_mean gauge",
        f"loadfc_latency_ms_mean {mean_latency:.3f}",
        "# HELP loadfc_model_loaded Whether a model artefact is in memory.",
        "# TYPE loadfc_model_loaded gauge",
        f"loadfc_model_loaded {1 if state.bundle else 0}",
    ]

    if state.monitor is not None and state.recent:
        report = state.monitor.score(pd.concat(list(state.recent)))
        worst = report.worst
        if worst is not None:
            lines += [
                "# HELP loadfc_drift_psi_max Largest feature PSI over recent requests.",
                "# TYPE loadfc_drift_psi_max gauge",
                f"loadfc_drift_psi_max {worst[1]:.6f}",
            ]

    return "\n".join(lines) + "\n"
