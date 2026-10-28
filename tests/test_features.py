"""Feature construction, with leakage as the primary concern."""

import numpy as np
import pandas as pd
import pytest

from loadfc.features import (
    FeatureSpec,
    LeakageError,
    build_features,
    build_inference_features,
    calendar_features,
)


def test_spec_rejects_lags_inside_the_horizon():
    spec = FeatureSpec(horizon=24, lag_offsets=(0,), absolute_lags=(1,))
    with pytest.raises(LeakageError, match="shorter than the 24 hour"):
        spec.validate()


def test_spec_accepts_a_lag_exactly_at_the_horizon():
    FeatureSpec(horizon=24, lag_offsets=(0,), absolute_lags=()).validate()


@pytest.mark.parametrize("horizon", [1, 6, 24, 48])
def test_every_lag_respects_the_horizon(horizon):
    spec = FeatureSpec(horizon=horizon)
    spec.validate()
    assert min(spec.lags) >= horizon


def test_lag_features_hold_the_value_from_that_many_hours_earlier(hourly_series):
    matrix = build_features(hourly_series, FeatureSpec(horizon=24))
    timestamp = matrix.index[5000]

    for lag in (24, 168, 336):
        expected = hourly_series.loc[timestamp - pd.Timedelta(hours=lag)]
        assert matrix.x.loc[timestamp, f"lag_{lag}h"] == pytest.approx(expected)


def test_no_feature_correlates_perfectly_with_the_target(hourly_series):
    """A near-perfect correlation at a 24 hour horizon means a leak."""
    matrix = build_features(hourly_series, FeatureSpec(horizon=24))
    correlations = matrix.x.corrwith(matrix.y).abs()
    assert correlations.max() < 0.99, correlations.sort_values().tail(3)


def test_rolling_windows_end_at_the_horizon(hourly_series):
    spec = FeatureSpec(horizon=24, rolling_windows=(24,))
    matrix = build_features(hourly_series, spec)
    timestamp = matrix.index[5000]

    window_end = timestamp - pd.Timedelta(hours=24)
    window_start = window_end - pd.Timedelta(hours=23)
    expected = hourly_series.loc[window_start:window_end].mean()
    assert matrix.x.loc[timestamp, "roll_mean_24h"] == pytest.approx(expected)


def test_features_require_a_regular_hourly_grid(hourly_series):
    gappy = hourly_series.drop(hourly_series.index[100:110])
    with pytest.raises(ValueError, match="complete hourly grid"):
        build_features(gappy)


def test_calendar_encodes_hours_cyclically():
    index = pd.date_range("2021-01-01", periods=48, freq="h")
    frame = calendar_features(index)

    # Hour 23 and hour 0 must be neighbours in the encoded space.
    distance_wrap = np.hypot(
        frame["hour_sin"].iloc[23] - frame["hour_sin"].iloc[24],
        frame["hour_cos"].iloc[23] - frame["hour_cos"].iloc[24],
    )
    distance_far = np.hypot(
        frame["hour_sin"].iloc[0] - frame["hour_sin"].iloc[12],
        frame["hour_cos"].iloc[0] - frame["hour_cos"].iloc[12],
    )
    assert distance_wrap < distance_far


def test_calendar_marks_a_known_federal_holiday():
    index = pd.date_range("2021-07-03", periods=72, freq="h")
    frame = calendar_features(index, holidays=True)
    july_fourth = frame.loc["2021-07-05", "is_holiday"]  # observed Monday
    assert july_fourth.max() == 1


def test_inference_features_cover_the_requested_steps(hourly_series):
    history = hourly_series.iloc[:1000]
    features = build_inference_features(history, FeatureSpec(horizon=24), steps=24)

    assert len(features) == 24
    assert features.index[0] == history.index[-1] + pd.Timedelta(hours=1)
    assert not features.isna().any().any()


def test_inference_refuses_to_forecast_past_the_horizon(hourly_series):
    with pytest.raises(LeakageError, match="longest honest forecast"):
        build_inference_features(hourly_series.iloc[:1000], FeatureSpec(horizon=24), steps=25)


def test_inference_requires_enough_history(hourly_series):
    with pytest.raises(ValueError, match="at least"):
        build_inference_features(hourly_series.iloc[:50], FeatureSpec(horizon=24), steps=24)


def test_training_and_inference_produce_the_same_columns(hourly_series):
    spec = FeatureSpec(horizon=24)
    training = build_features(hourly_series, spec)
    inference = build_inference_features(hourly_series.iloc[:1000], spec, steps=12)
    assert list(training.x.columns) == list(inference.columns)
