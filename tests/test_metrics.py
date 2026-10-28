"""Error metrics, especially what MASE means."""

import numpy as np
import pandas as pd
import pytest

from loadfc.metrics import evaluate, mae, mape, mase, peak_hour_mae, rmse, seasonal_naive_scale


def test_perfect_forecast_scores_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert mape(y, y) == 0.0


def test_rmse_punishes_outliers_more_than_mae():
    y_true = np.zeros(100)
    y_pred = np.zeros(100)
    y_pred[0] = 50.0
    assert rmse(y_true, y_pred) > mae(y_true, y_pred)


def test_mape_is_a_percentage():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    assert mape(y_true, y_pred) == pytest.approx(10.0)


def test_mase_of_the_seasonal_naive_forecast_is_about_one():
    """The definition made concrete: scaling by seasonal naive means it scores ~1."""
    rng = np.random.default_rng(0)
    period = 168
    index = np.arange(period * 12)
    series = 100 + 20 * np.sin(2 * np.pi * index / period) + rng.normal(0, 2, len(index))

    train, test = series[: period * 8], series[period * 8 :]
    naive = series[period * 8 - period : -period]
    assert mase(test, naive, train, period) == pytest.approx(1.0, abs=0.15)


def test_mase_below_one_means_better_than_naive():
    rng = np.random.default_rng(1)
    train = rng.normal(100, 10, 2000)
    y_true = rng.normal(100, 10, 500)
    good = y_true + rng.normal(0, 0.1, 500)
    assert mase(y_true, good, train) < 1.0


def test_seasonal_scale_rejects_a_series_shorter_than_one_period():
    with pytest.raises(ValueError, match="shorter than one seasonal period"):
        seasonal_naive_scale(np.arange(10.0), period=168)


def test_peak_hour_error_only_counts_peak_hours():
    index = pd.date_range("2021-01-01", periods=24, freq="h")
    y_true = np.zeros(24)
    y_pred = np.zeros(24)
    y_pred[18] = 100.0  # inside the 17-20 peak window

    assert peak_hour_mae(y_true, y_pred, index) == pytest.approx(25.0)
    assert peak_hour_mae(y_true, y_pred, index, hours=(3,)) == 0.0


def test_evaluate_returns_every_metric():
    rng = np.random.default_rng(2)
    index = pd.date_range("2021-01-01", periods=500, freq="h")
    y_true = rng.normal(100, 10, 500)
    scores = evaluate(y_true, y_true + 1.0, rng.normal(100, 10, 2000), index)
    assert set(scores) >= {"mae", "rmse", "mape", "mase", "peak_hour_mae", "n"}
