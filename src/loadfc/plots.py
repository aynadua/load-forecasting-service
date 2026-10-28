"""Figures for the README. Headless by construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def load_profile(series: pd.Series, path: str | Path) -> Path:
    """The seasonality the model has to capture, at three time scales."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), dpi=140)

    hourly = series.groupby(series.index.hour).mean()
    axes[0].plot(hourly.index, hourly.to_numpy(), marker="o", markersize=3)
    axes[0].set_title("Daily shape", fontsize=10)
    axes[0].set_xlabel("hour of day")

    weekly = series.groupby(series.index.day_of_week).mean()
    axes[1].bar(weekly.index, weekly.to_numpy())
    axes[1].set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[1].set_title("Weekly shape", fontsize=10)

    monthly = series.groupby(series.index.month).mean()
    axes[2].plot(monthly.index, monthly.to_numpy(), marker="o", markersize=3)
    axes[2].set_title("Annual shape (summer and winter peaks)", fontsize=10)
    axes[2].set_xlabel("month")

    for ax in axes:
        ax.set_ylabel("mean load (MW)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def fold_scores(record: dict[str, Any], path: str | Path, metric: str = "mape") -> Path:
    """Per-fold error for every model: the spread matters as much as the mean."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    per_fold = record["per_fold"]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
    positions = np.arange(len(record["folds"]))
    width = 0.8 / max(len(per_fold), 1)

    for i, (name, scores) in enumerate(per_fold.items()):
        ax.bar(positions + i * width, [s[metric] for s in scores], width, label=name)

    labels = [f"{f['test_start'][:7]}\nto {f['test_end'][:7]}" for f in record["folds"]]
    ax.set_xticks(positions + width * (len(per_fold) - 1) / 2, labels, fontsize=7)
    ax.set_ylabel(metric.upper() + (" (%)" if metric == "mape" else " (MW)"))
    ax.set_title(f"{metric.upper()} by rolling-origin fold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def forecast_window(
    actual: pd.Series, predictions: dict[str, pd.Series], path: str | Path, hours: int = 336
) -> Path:
    """Two weeks of forecast against actuals, so the shape is visible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    window = actual.iloc[-hours:]
    fig, ax = plt.subplots(figsize=(12, 4), dpi=140)
    ax.plot(window.index, window.to_numpy(), label="actual", linewidth=1.6, color="black")
    for name, prediction in predictions.items():
        aligned = prediction.reindex(window.index)
        ax.plot(aligned.index, aligned.to_numpy(), linewidth=1.0, alpha=0.85, label=name)

    ax.set_ylabel("load (MW)")
    ax.set_title(f"Last {hours // 24} days of the final backtest fold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def error_by_hour(
    actual: pd.Series, predictions: dict[str, pd.Series], path: str | Path
) -> Path:
    """Where in the day each model is weak."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    for name, prediction in predictions.items():
        aligned = prediction.reindex(actual.index)
        error = (actual - aligned).abs()
        ax.plot(error.groupby(error.index.hour).mean(), marker="o", markersize=3, label=name)

    ax.set_xlabel("hour of day")
    ax.set_ylabel("mean absolute error (MW)")
    ax.set_title("Error by hour of day (evening peak is hours 17-20)")
    ax.axvspan(17, 20, alpha=0.12, color="tab:red")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def importance(series: pd.Series, path: str | Path, top: int = 15) -> Path:
    """Permutation importance, in MW of added error."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    head = series.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    ax.barh(head.index, head.to_numpy())
    ax.set_xlabel("increase in MAE when shuffled (MW)")
    ax.set_title("Permutation importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
