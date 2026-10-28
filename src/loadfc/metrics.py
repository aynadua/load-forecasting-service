"""Forecast error metrics.

MAPE is reported because the electricity industry reports it, but it is not
the metric to judge on: it is asymmetric and it inflates when load is low.
MASE is the honest one. It divides error by the error of a seasonal naive
forecast on the *training* data, so a MASE below 1 means the model genuinely
beats "same hour last week" and a MASE above 1 means it does not, whatever the
MAPE says.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEKLY_PERIOD = 168


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, in percent."""
    nonzero = y_true != 0
    return float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100.0)


def seasonal_naive_scale(y_train: np.ndarray, period: int = WEEKLY_PERIOD) -> float:
    """Mean absolute error of a seasonal naive forecast on the training data."""
    if len(y_train) <= period:
        raise ValueError("training series is shorter than one seasonal period")
    return float(np.mean(np.abs(y_train[period:] - y_train[:-period])))


def mase(
    y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, period: int = WEEKLY_PERIOD
) -> float:
    """Mean absolute scaled error. Below 1.0 beats seasonal naive."""
    scale = seasonal_naive_scale(y_train, period)
    return mae(y_true, y_pred) / scale if scale > 0 else float("inf")


def peak_hour_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    index: pd.DatetimeIndex,
    hours: tuple[int, ...] = (17, 18, 19, 20),
) -> float:
    """Error restricted to the evening peak.

    Grid operators care disproportionately about the peak, because that is
    what reserve margins and spot prices are set against. A model that is
    accurate on average and poor at 6pm is not a useful model.
    """
    mask = np.isin(index.hour, hours)
    return mae(y_true[mask], y_pred[mask]) if mask.any() else float("nan")


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    index: pd.DatetimeIndex | None = None,
) -> dict[str, float]:
    """Every metric at once, for one fold or one whole backtest."""
    scores = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "mase": mase(y_true, y_pred, y_train),
        "n": int(len(y_true)),
    }
    if index is not None:
        scores["peak_hour_mae"] = peak_hour_mae(y_true, y_pred, index)
    return scores
