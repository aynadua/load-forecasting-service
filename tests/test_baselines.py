"""Seasonal naive baselines and the horizon guard that keeps them honest."""

import numpy as np
import pandas as pd
import pytest

from loadfc.baselines import BASELINES, SeasonalNaive, build


def test_prediction_is_the_value_one_period_earlier(hourly_series):
    model = SeasonalNaive(period_hours=168).fit(hourly_series)
    target = hourly_series.index[5000:5024]
    predictions = model.predict(target)

    expected = hourly_series.loc[target - pd.Timedelta(hours=168)].to_numpy()
    assert np.allclose(predictions, expected)


def test_a_baseline_shorter_than_the_horizon_is_refused():
    with pytest.raises(ValueError, match="inside the"):
        build("seasonal_naive_daily").validate_for_horizon(48)


def test_a_baseline_exactly_at_the_horizon_is_allowed():
    build("seasonal_naive_daily").validate_for_horizon(24)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="call fit"):
        SeasonalNaive().predict(pd.date_range("2021-01-01", periods=3, freq="h"))


def test_unknown_baseline_name_is_rejected():
    with pytest.raises(ValueError, match="unknown baseline"):
        build("crystal_ball")


def test_missing_history_falls_back_to_the_mean():
    index = pd.date_range("2021-01-01", periods=200, freq="h")
    series = pd.Series(np.arange(200, dtype=float), index=index)
    model = SeasonalNaive(period_hours=168).fit(series)

    # Targets whose lookback predates the series.
    early = index[:10]
    assert np.allclose(model.predict(early), series.mean())


@pytest.mark.parametrize("name", BASELINES)
def test_every_registered_baseline_builds(name):
    assert build(name).period_hours > 0
