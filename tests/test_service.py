"""Service behaviour, especially what it refuses to do.

A forecasting API that silently produces numbers from malformed or
insufficient input is worse than one that errors, because the caller cannot
tell the difference. Most of these tests are about the refusals.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from loadfc import service as service_module
from loadfc.drift import DriftMonitor
from loadfc.features import FeatureSpec, build_features
from loadfc.model import fit


@pytest.fixture
def client(hourly_series, monkeypatch):
    """A client backed by a small model fitted on the synthetic series."""
    spec = FeatureSpec(horizon=24)
    matrix = build_features(hourly_series, spec)
    bundle = fit(matrix, seed=0, max_iter=25, early_stopping=False)

    monkeypatch.setattr(service_module.state, "bundle", bundle)
    monkeypatch.setattr(
        service_module.state, "monitor", DriftMonitor.from_training(matrix.x, sample=500)
    )
    service_module.state.recent.clear()
    service_module.state.counters.update(
        {"requests_total": 0.0, "errors_total": 0.0,
         "forecast_hours_total": 0.0, "latency_ms_sum": 0.0}
    )
    return TestClient(service_module.app)


def _history(series: pd.Series, hours: int = 600) -> list[dict]:
    window = series.iloc[-hours:]
    return [
        {"timestamp": str(ts), "load_mw": float(v)}
        for ts, v in zip(window.index, window, strict=True)
    ]


def test_health_reports_a_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["horizon_hours"] == 24


def test_forecast_returns_the_requested_hours(client, hourly_series):
    response = client.post(
        "/forecast", json={"history": _history(hourly_series), "steps": 24}
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["forecast"]) == 24
    assert body["horizon_hours"] == 24
    assert all(np.isfinite(point["load_mw"]) for point in body["forecast"])


def test_forecast_timestamps_follow_the_history(client, hourly_series):
    history = _history(hourly_series)
    body = client.post("/forecast", json={"history": history, "steps": 3}).json()

    last = pd.Timestamp(history[-1]["timestamp"])
    expected = [str(last + pd.Timedelta(hours=i)) for i in (1, 2, 3)]
    assert [point["timestamp"] for point in body["forecast"]] == expected


def test_forecasts_are_physically_plausible(client, hourly_series):
    body = client.post(
        "/forecast", json={"history": _history(hourly_series), "steps": 24}
    ).json()
    values = [point["load_mw"] for point in body["forecast"]]
    assert min(values) > 0.5 * hourly_series.min()
    assert max(values) < 1.5 * hourly_series.max()


def test_too_little_history_is_refused(client, hourly_series):
    response = client.post(
        "/forecast", json={"history": _history(hourly_series, hours=50), "steps": 24}
    )
    assert response.status_code == 422
    assert "history" in response.json()["detail"]


def test_a_gap_in_the_history_is_refused(client, hourly_series):
    history = _history(hourly_series)
    del history[100]
    response = client.post("/forecast", json={"history": history, "steps": 24})
    assert response.status_code == 422
    assert "contiguous" in response.json()["detail"]


def test_duplicate_timestamps_are_refused(client, hourly_series):
    history = _history(hourly_series)
    history.append(dict(history[-1]))
    response = client.post("/forecast", json={"history": history, "steps": 24})
    assert response.status_code == 422
    assert "duplicate" in response.json()["detail"]


def test_asking_beyond_the_horizon_is_refused_by_validation(client, hourly_series):
    response = client.post(
        "/forecast", json={"history": _history(hourly_series), "steps": 48}
    )
    assert response.status_code == 422


def test_negative_load_is_refused(client, hourly_series):
    history = _history(hourly_series)
    history[0]["load_mw"] = -5.0
    assert client.post("/forecast", json={"history": history, "steps": 24}).status_code == 422


def test_metrics_are_prometheus_formatted(client, hourly_series):
    client.post("/forecast", json={"history": _history(hourly_series), "steps": 24})
    text = client.get("/metrics").text

    assert "# TYPE loadfc_requests_total counter" in text
    assert "loadfc_requests_total 1" in text
    assert "loadfc_forecast_hours_total 24" in text
    assert "loadfc_model_loaded 1" in text


def test_drift_endpoint_scores_served_requests(client, hourly_series):
    assert client.get("/drift").json()["status"] == "no_data"

    client.post("/forecast", json={"history": _history(hourly_series), "steps": 24})
    body = client.get("/drift").json()
    assert body["status"] in {"stable", "moderate", "significant"}
    assert body["n_live"] == 24


def test_service_reports_degraded_without_a_model(monkeypatch):
    monkeypatch.setattr(service_module.state, "bundle", None)
    client = TestClient(service_module.app)
    assert client.get("/health").json()["status"] == "degraded"
    response = client.post(
        "/forecast", json={"history": [{"timestamp": "2020-01-01", "load_mw": 1.0}]}
    )
    assert response.status_code == 503
