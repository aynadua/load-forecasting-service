"""Rolling-origin backtesting.

A single train/test split of a time series gives one number that depends
entirely on which year happened to land in the test set. Load changes: demand
grew, then flattened as efficiency and rooftop solar spread. A model evaluated
only on 2016 tells you about 2016.

So evaluation is rolling-origin with an expanding window: train on everything
before a cut-off, test on the window after it, advance, repeat. Every fold
trains a fresh model, nothing from a fold's test window reaches its training
set, and the spread across folds is reported alongside the mean -- because
that spread is the honest measure of how much the headline number can be
trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from loadfc import model as model_module
from loadfc.baselines import BASELINES, build
from loadfc.features import FeatureSpec, build_features
from loadfc.metrics import evaluate


@dataclass
class Fold:
    """One train/test cut of the series."""

    index: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class BacktestResult:
    """Per-fold scores for every model, plus the pooled predictions."""

    folds: list[Fold]
    scores: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    predictions: dict[str, pd.Series] = field(default_factory=dict)
    actuals: pd.Series | None = None

    def summary(self) -> dict[str, dict[str, float]]:
        """Mean and spread of each metric across folds, per model."""
        out: dict[str, dict[str, float]] = {}
        for name, fold_scores in self.scores.items():
            entry: dict[str, float] = {}
            for metric in ("mae", "rmse", "mape", "mase", "peak_hour_mae"):
                values = [f[metric] for f in fold_scores if metric in f]
                if values:
                    entry[metric] = float(np.mean(values))
                    entry[f"{metric}_std"] = float(np.std(values))
                    entry[f"{metric}_worst"] = float(np.max(values))
            entry["n_folds"] = len(fold_scores)
            out[name] = entry
        return out


def make_folds(
    index: pd.DatetimeIndex, n_folds: int = 6, test_hours: int = 8760, min_train_hours: int = 26280
) -> list[Fold]:
    """Expanding-window folds, each testing on ``test_hours`` of unseen data.

    Defaults give six one-year test windows after at least three years of
    training history.
    """
    if len(index) < min_train_hours + n_folds * test_hours:
        raise ValueError(
            f"series of {len(index)} hours cannot support {n_folds} folds of "
            f"{test_hours} hours after {min_train_hours} hours of training history"
        )

    folds: list[Fold] = []
    end = len(index)
    for i in range(n_folds):
        test_stop = end - (n_folds - 1 - i) * test_hours
        test_start = test_stop - test_hours
        folds.append(
            Fold(
                index=i,
                train_end=index[test_start - 1],
                test_start=index[test_start],
                test_end=index[test_stop - 1],
            )
        )
    return folds


def run(
    series: pd.Series,
    spec: FeatureSpec | None = None,
    n_folds: int = 6,
    test_hours: int = 8760,
    min_train_hours: int = 26280,
    seed: int = 7,
    verbose: bool = True,
) -> BacktestResult:
    """Backtest the model and every baseline over rolling origins."""
    spec = spec or FeatureSpec()
    matrix = build_features(series, spec)
    folds = make_folds(
        matrix.index, n_folds=n_folds, test_hours=test_hours, min_train_hours=min_train_hours
    )

    result = BacktestResult(folds=folds)
    collected: dict[str, list[pd.Series]] = {}
    actuals: list[pd.Series] = []

    for fold in folds:
        train_mask = matrix.index <= fold.train_end
        test_mask = (matrix.index >= fold.test_start) & (matrix.index <= fold.test_end)

        x_train, y_train = matrix.x[train_mask], matrix.y[train_mask]
        x_test, y_test = matrix.x[test_mask], matrix.y[test_mask]
        actuals.append(y_test)

        fold_matrix = type(matrix)(x=x_train, y=y_train, spec=spec)
        bundle = model_module.fit(fold_matrix, seed=seed)
        predictions = {"gradient boosting": pd.Series(bundle.predict(x_test), index=y_test.index)}

        for name in BASELINES:
            baseline = build(name)
            baseline.validate_for_horizon(spec.horizon)
            # Observed history only: everything strictly before the test window,
            # plus the test window's own past, which the lag keeps legal.
            baseline.fit(series[series.index <= fold.test_end])
            predictions[baseline.name] = pd.Series(
                baseline.predict(y_test.index), index=y_test.index
            )

        for name, prediction in predictions.items():
            scores = evaluate(
                y_test.to_numpy(), prediction.to_numpy(), y_train.to_numpy(), y_test.index
            )
            result.scores.setdefault(name, []).append(scores)
            collected.setdefault(name, []).append(prediction)

        if verbose:
            headline = {n: f"{result.scores[n][-1]['mape']:.2f}%" for n in predictions}
            print(
                f"fold {fold.index}: train<={fold.train_end:%Y-%m-%d} "
                f"test {fold.test_start:%Y-%m-%d}..{fold.test_end:%Y-%m-%d}  {headline}",
                flush=True,
            )

    result.actuals = pd.concat(actuals)
    result.predictions = {name: pd.concat(parts) for name, parts in collected.items()}
    return result


def to_record(result: BacktestResult, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """JSON-serialisable summary for reports/."""
    return {
        "summary": result.summary(),
        "folds": [
            {
                "index": f.index,
                "train_end": str(f.train_end),
                "test_start": str(f.test_start),
                "test_end": str(f.test_end),
            }
            for f in result.folds
        ],
        "per_fold": result.scores,
        **(extra or {}),
    }
