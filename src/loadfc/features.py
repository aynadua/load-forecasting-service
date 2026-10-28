"""Feature construction, built around one rule: nothing from the future.

A day-ahead forecast is issued before the day starts. At the moment of issue
the most recent *observed* load is at least ``horizon`` hours old. So every
feature derived from the target series must be lagged by at least that much.

This is the single most common defect in published load-forecasting code. A
model given ``lag_1`` to predict 24 hours ahead reports a spectacular MAPE and
cannot be deployed, because at run time that value does not exist yet. Here
:func:`build_features` refuses to construct such a feature at all --
:class:`LeakageError` is raised rather than a warning logged, because a silent
version of this mistake is worth more in benchmark scores than any amount of
modelling.

Calendar features are exempt and correctly so: the hour of day and the date of
a public holiday are known years in advance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

TARGET = "load_mw"


class LeakageError(ValueError):
    """Raised when a requested feature would need data the forecaster cannot have."""


@dataclass
class FeatureSpec:
    """Which lags and rolling windows to build, relative to the forecast horizon.

    Offsets are *extra* hours beyond the horizon, so the same spec stays valid
    at any horizon. ``lag_offsets = [0, 1, 2]`` at a 24 hour horizon means lags
    of 24, 25 and 26 hours.
    """

    horizon: int = 24
    lag_offsets: tuple[int, ...] = (0, 1, 2, 23, 24, 25)
    absolute_lags: tuple[int, ...] = (168, 336)
    rolling_windows: tuple[int, ...] = (24, 168)
    calendar: bool = True
    holidays: bool = True

    @property
    def lags(self) -> list[int]:
        """Every lag this spec implies, in hours, sorted."""
        relative = [self.horizon + offset for offset in self.lag_offsets]
        return sorted(set(relative) | set(self.absolute_lags))

    def validate(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be at least one hour")
        unusable = [lag for lag in self.lags if lag < self.horizon]
        if unusable:
            raise LeakageError(
                f"lags {unusable} are shorter than the {self.horizon} hour forecast horizon; "
                "those values are not observable when the forecast is issued"
            )


@dataclass
class FeatureMatrix:
    """Aligned design matrix, target and timestamps, with no missing rows."""

    x: pd.DataFrame
    y: pd.Series
    spec: FeatureSpec = field(repr=False)

    def __len__(self) -> int:
        return len(self.y)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.y.index


def _assert_regular_hourly(series: pd.Series) -> None:
    """Positional shifts only equal time shifts on a complete hourly grid."""
    if len(series) < 2:
        raise ValueError("series is too short")
    deltas = series.index.to_series().diff().dropna().unique()
    if len(deltas) != 1 or deltas[0] != pd.Timedelta(hours=1):
        raise ValueError(
            "series must sit on a complete hourly grid before features are built; "
            "run loadfc.data.clean first"
        )


def calendar_features(index: pd.DatetimeIndex, holidays: bool = True) -> pd.DataFrame:
    """Time-of-day, time-of-week and time-of-year effects, known in advance.

    Cyclical quantities are encoded as sine/cosine pairs so that hour 23 is
    adjacent to hour 0 rather than 23 units away from it.
    """
    frame = pd.DataFrame(index=index)

    for name, values, period in (
        ("hour", index.hour, 24),
        ("dayofweek", index.day_of_week, 7),
        ("month", index.month - 1, 12),
        ("dayofyear", index.day_of_year - 1, 365.25),
    ):
        angle = 2.0 * np.pi * np.asarray(values) / period
        frame[f"{name}_sin"] = np.sin(angle)
        frame[f"{name}_cos"] = np.cos(angle)

    frame["is_weekend"] = (index.day_of_week >= 5).astype(np.int8)

    if holidays:
        calendar = USFederalHolidayCalendar()
        dates = calendar.holidays(start=index.min(), end=index.max())
        frame["is_holiday"] = index.normalize().isin(dates).astype(np.int8)

    return frame


def build_features(series: pd.Series, spec: FeatureSpec | None = None) -> FeatureMatrix:
    """Build a leakage-free design matrix for forecasting ``spec.horizon`` ahead."""
    spec = spec or FeatureSpec()
    spec.validate()
    _assert_regular_hourly(series)

    frame = pd.DataFrame(index=series.index)

    for lag in spec.lags:
        frame[f"lag_{lag}h"] = series.shift(lag)

    # Rolling statistics over a window that ends at the most recent observable
    # point, which is `horizon` hours before the target.
    observable = series.shift(spec.horizon)
    for window in spec.rolling_windows:
        frame[f"roll_mean_{window}h"] = observable.rolling(window).mean()
        frame[f"roll_std_{window}h"] = observable.rolling(window).std()
        frame[f"roll_min_{window}h"] = observable.rolling(window).min()
        frame[f"roll_max_{window}h"] = observable.rolling(window).max()

    # Week-on-week change at the same hour: captures trend without using
    # anything newer than the horizon allows.
    if 168 in spec.lags and 336 in spec.lags:
        frame["weekly_delta"] = frame["lag_168h"] - frame["lag_336h"]

    if spec.calendar:
        frame = frame.join(calendar_features(series.index, spec.holidays))

    target = series.rename(TARGET)
    combined = frame.join(target).dropna()
    return FeatureMatrix(
        x=combined.drop(columns=[TARGET]), y=combined[TARGET], spec=spec
    )


def feature_names(spec: FeatureSpec | None = None) -> list[str]:
    """Column order the model expects, without needing the data."""
    spec = spec or FeatureSpec()
    span = max(spec.lags) + max(spec.rolling_windows) + 48
    index = pd.date_range("2020-01-01", periods=span, freq="h")
    dummy = pd.Series(np.arange(len(index), dtype=float), index=index)
    return list(build_features(dummy, spec).x.columns)


def build_inference_features(
    history: pd.Series, spec: FeatureSpec | None = None, steps: int | None = None
) -> pd.DataFrame:
    """Design matrix for the next ``steps`` hours after the end of ``history``.

    ``steps`` defaults to the horizon and may not exceed it. The reason is the
    same rule the training features obey: forecasting ``horizon + 1`` hours
    ahead from this history would need an observation one hour after the last
    one, which does not exist yet. Asking for it raises rather than silently
    filling a lag with NaN and returning a confident-looking number.
    """
    spec = spec or FeatureSpec()
    spec.validate()
    _assert_regular_hourly(history)

    steps = spec.horizon if steps is None else steps
    if not 1 <= steps <= spec.horizon:
        raise LeakageError(
            f"cannot forecast {steps} hours from a model trained at a {spec.horizon} hour "
            f"horizon; the longest honest forecast from this history is {spec.horizon} hours"
        )

    required = max(spec.lags) + max(spec.rolling_windows, default=0)
    if len(history) < required:
        raise ValueError(
            f"need at least {required} hours of history for these features, got {len(history)}"
        )

    future = pd.date_range(
        history.index.max() + pd.Timedelta(hours=1), periods=steps, freq="h"
    )
    extended = pd.concat([history, pd.Series(np.nan, index=future)])

    frame = pd.DataFrame(index=extended.index)
    for lag in spec.lags:
        frame[f"lag_{lag}h"] = extended.shift(lag)

    observable = extended.shift(spec.horizon)
    for window in spec.rolling_windows:
        frame[f"roll_mean_{window}h"] = observable.rolling(window).mean()
        frame[f"roll_std_{window}h"] = observable.rolling(window).std()
        frame[f"roll_min_{window}h"] = observable.rolling(window).min()
        frame[f"roll_max_{window}h"] = observable.rolling(window).max()

    if 168 in spec.lags and 336 in spec.lags:
        frame["weekly_delta"] = frame["lag_168h"] - frame["lag_336h"]

    if spec.calendar:
        frame = frame.join(calendar_features(extended.index, spec.holidays))

    out = frame.loc[future]
    if out.isna().any().any():
        incomplete = sorted(out.columns[out.isna().any()])
        raise ValueError(f"history leaves these features undefined: {incomplete}")
    return out
