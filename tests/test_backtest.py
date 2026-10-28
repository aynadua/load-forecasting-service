"""Backtest mechanics.

:func:`test_no_fold_trains_on_its_own_test_window` and
:func:`test_folds_move_forward_in_time` are the ones that matter. A backtest
that quietly trains on future data produces beautiful numbers and a worthless
model, and the defect is invisible in the output.
"""

import pandas as pd
import pytest

from loadfc.backtest import BacktestResult, make_folds, run, to_record
from loadfc.features import FeatureSpec


def test_folds_move_forward_in_time(hourly_series):
    folds = make_folds(hourly_series.index, n_folds=3, test_hours=1000, min_train_hours=2000)
    assert len(folds) == 3
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert later.test_start > earlier.test_start
        assert later.train_end >= earlier.test_end


def test_no_fold_trains_on_its_own_test_window(hourly_series):
    folds = make_folds(hourly_series.index, n_folds=3, test_hours=1000, min_train_hours=2000)
    for fold in folds:
        assert fold.train_end < fold.test_start


def test_test_windows_do_not_overlap(hourly_series):
    folds = make_folds(hourly_series.index, n_folds=4, test_hours=800, min_train_hours=2000)
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert earlier.test_end < later.test_start


def test_each_test_window_has_the_requested_length(hourly_series):
    folds = make_folds(hourly_series.index, n_folds=2, test_hours=720, min_train_hours=2000)
    for fold in folds:
        span = (fold.test_end - fold.test_start) / pd.Timedelta(hours=1)
        assert span == pytest.approx(719)


def test_too_few_hours_is_refused(hourly_series):
    with pytest.raises(ValueError, match="cannot support"):
        make_folds(hourly_series.index[:500], n_folds=6, test_hours=8760)


def test_run_scores_the_model_and_both_baselines(hourly_series):
    result = run(
        hourly_series,
        FeatureSpec(horizon=24),
        n_folds=2,
        test_hours=720,
        min_train_hours=2000,
        verbose=False,
    )
    assert len(result.folds) == 2
    assert "gradient boosting" in result.scores
    assert len([k for k in result.scores if "naive" in k]) == 2
    for scores in result.scores.values():
        assert len(scores) == 2


def test_summary_reports_mean_and_spread(hourly_series):
    result = run(
        hourly_series,
        FeatureSpec(horizon=24),
        n_folds=2,
        test_hours=720,
        min_train_hours=2000,
        verbose=False,
    )
    summary = result.summary()
    for scores in summary.values():
        assert "mape" in scores and "mape_std" in scores and "mape_worst" in scores
        assert scores["n_folds"] == 2


def test_predictions_align_with_actuals(hourly_series):
    result = run(
        hourly_series,
        FeatureSpec(horizon=24),
        n_folds=2,
        test_hours=720,
        min_train_hours=2000,
        verbose=False,
    )
    for prediction in result.predictions.values():
        assert prediction.index.equals(result.actuals.index)


def test_record_is_json_serialisable(hourly_series):
    import json

    result = run(
        hourly_series,
        FeatureSpec(horizon=24),
        n_folds=2,
        test_hours=720,
        min_train_hours=2000,
        verbose=False,
    )
    json.dumps(to_record(result))


def test_empty_result_summarises_without_error():
    assert BacktestResult(folds=[]).summary() == {}
