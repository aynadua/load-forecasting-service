"""Cleaning, with attention to the daylight-saving artefacts in the real file."""

import pandas as pd
import pytest

from loadfc.data import TARGET, clean, load_raw, seasonal_strength, summarise


def test_duplicate_timestamps_are_averaged(raw_frame_with_dst):
    frame = pd.DataFrame(
        {"timestamp": raw_frame_with_dst["Datetime"], TARGET: raw_frame_with_dst["PJME_MW"]}
    )
    series, report = clean(frame)

    assert report.n_duplicate_timestamps == 1
    assert not series.index.has_duplicates


def test_missing_hours_are_restored_on_the_grid(raw_frame_with_dst):
    frame = pd.DataFrame(
        {"timestamp": raw_frame_with_dst["Datetime"], TARGET: raw_frame_with_dst["PJME_MW"]}
    )
    series, report = clean(frame)

    assert report.n_missing_hours >= 1
    assert pd.Timestamp("2020-03-02 02:00") in series.index
    deltas = series.index.to_series().diff().dropna().unique()
    assert len(deltas) == 1 and deltas[0] == pd.Timedelta(hours=1)


def test_long_gaps_are_left_missing_rather_than_invented():
    index = pd.date_range("2020-01-01", periods=200, freq="h")
    values = pd.Series(range(200), index=index, dtype=float)
    # A ten-hour outage, far longer than the interpolation limit.
    kept = values.drop(values.index[50:60])

    frame = pd.DataFrame({"timestamp": kept.index, TARGET: kept.to_numpy()})
    series, report = clean(frame, max_interpolate_hours=3)

    assert report.n_left_missing == 10
    assert series.isna().sum() == 10


def test_short_gaps_are_interpolated():
    index = pd.date_range("2020-01-01", periods=100, freq="h")
    values = pd.Series(range(100), index=index, dtype=float)
    kept = values.drop(values.index[40:42])

    frame = pd.DataFrame({"timestamp": kept.index, TARGET: kept.to_numpy()})
    series, report = clean(frame, max_interpolate_hours=3)

    assert report.n_interpolated == 2
    assert series.isna().sum() == 0
    assert series.loc[pd.Timestamp("2020-01-02 16:00")] == pytest.approx(40.0)


def test_summarise_reports_the_span(hourly_series):
    stats = summarise(hourly_series)
    assert stats["n_hours"] == len(hourly_series)
    assert stats["min_mw"] < stats["mean_mw"] < stats["max_mw"]


def test_seasonal_strength_detects_the_daily_cycle(hourly_series):
    """A series built with a strong daily cycle must register one."""
    assert seasonal_strength(hourly_series, 24) > 0.2


def test_load_raw_rejects_an_unexpected_schema(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="unexpected columns"):
        load_raw(path)
