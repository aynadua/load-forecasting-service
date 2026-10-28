"""Shared fixtures: a synthetic but realistically shaped load series."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def hourly_series() -> pd.Series:
    """Three years of hourly load with daily, weekly and annual structure."""
    index = pd.date_range("2015-01-01", periods=24 * 365 * 3, freq="h")
    hours = np.arange(len(index))

    daily = 7000.0 * np.sin(2 * np.pi * (index.hour - 4) / 24)
    weekly = 2000.0 * (index.day_of_week < 5)
    annual = 5000.0 * np.cos(2 * np.pi * index.day_of_year / 365.25)
    noise = np.random.default_rng(0).normal(0, 600, len(index))

    values = 32000.0 + daily + weekly + annual + noise + 0.02 * hours
    series = pd.Series(values, index=index, name="load_mw")
    series.index.name = "timestamp"
    return series


@pytest.fixture
def raw_frame_with_dst() -> pd.DataFrame:
    """A small frame carrying both DST artefacts: a duplicate and a gap."""
    stamps = pd.date_range("2020-03-01", periods=48, freq="h").tolist()
    values = list(np.linspace(30000, 34000, 48))

    # Fall-back: 2020-03-01 05:00 recorded twice.
    stamps.append(pd.Timestamp("2020-03-01 05:00"))
    values.append(31000.0)

    frame = pd.DataFrame({"Datetime": stamps, "PJME_MW": values})
    # Spring-forward: drop 2020-03-02 02:00 entirely.
    return frame[frame["Datetime"] != pd.Timestamp("2020-03-02 02:00")].reset_index(drop=True)
