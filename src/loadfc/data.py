"""Loading and cleaning PJM East hourly system load.

The data is real: hourly metered load for the PJM East zone, roughly 2002 to
2018, in megawatts. Two things in it need handling before anything else, and
both come from the timestamps being naive *local* time across daylight-saving
transitions:

* **Spring forward.** The 02:00 hour does not exist, so the series has gaps.
* **Fall back.** The 02:00 hour happens twice, so some timestamps are
  duplicated.

Ignoring either one quietly corrupts every lag feature downstream: a lag of
"24 hours" computed by row offset stops being 24 hours the moment a row is
missing or doubled. So the series is reindexed onto a complete hourly grid and
lags are taken by *timestamp*, never by row position.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_URL = (
    "https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/"
    "master/data/PJME_hourly.csv"
)
TARGET = "load_mw"
TIMESTAMP = "timestamp"


@dataclass
class CleaningReport:
    """What cleaning had to do. Worth printing: silence here hides corruption."""

    n_raw: int
    n_duplicate_timestamps: int
    n_missing_hours: int
    n_interpolated: int
    n_left_missing: int
    first: pd.Timestamp
    last: pd.Timestamp

    def describe(self) -> str:
        return (
            f"{self.n_raw} raw rows spanning {self.first:%Y-%m-%d} to {self.last:%Y-%m-%d}; "
            f"{self.n_duplicate_timestamps} duplicate timestamps averaged (DST fall-back), "
            f"{self.n_missing_hours} hours absent from the grid, "
            f"{self.n_interpolated} interpolated, {self.n_left_missing} left missing"
        )


def download(destination: str | Path, url: str = SOURCE_URL) -> Path:
    """Fetch the raw CSV if it is not already on disk."""
    destination = Path(destination)
    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(url)
    frame.to_csv(destination, index=False)
    return destination


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the CSV into a tidy two-column frame."""
    frame = pd.read_csv(path)
    columns = {c.lower(): c for c in frame.columns}
    time_column = columns.get("datetime")
    value_column = next((c for c in frame.columns if c.upper().endswith("_MW")), None)
    if time_column is None or value_column is None:
        raise ValueError(f"unexpected columns {list(frame.columns)}")

    return pd.DataFrame(
        {
            TIMESTAMP: pd.to_datetime(frame[time_column]),
            TARGET: pd.to_numeric(frame[value_column], errors="coerce"),
        }
    )


def clean(frame: pd.DataFrame, max_interpolate_hours: int = 3) -> tuple[pd.Series, CleaningReport]:
    """Return an hourly series on a complete grid, plus a report of what changed.

    Gaps up to ``max_interpolate_hours`` are filled by time interpolation --
    load is smooth at that scale. Longer gaps are left as NaN rather than
    invented, and the feature builder drops any row that depends on them.
    """
    n_raw = len(frame)
    first, last = frame[TIMESTAMP].min(), frame[TIMESTAMP].max()

    duplicates = int(frame[TIMESTAMP].duplicated().sum())
    collapsed = frame.groupby(TIMESTAMP, as_index=True)[TARGET].mean().sort_index()

    grid = pd.date_range(collapsed.index.min(), collapsed.index.max(), freq="h")
    reindexed = collapsed.reindex(grid)
    n_missing = int(reindexed.isna().sum())

    filled = _interpolate_short_runs(reindexed, max_interpolate_hours)
    n_left = int(filled.isna().sum())

    filled.index.name = TIMESTAMP
    filled.name = TARGET
    return filled, CleaningReport(
        n_raw=n_raw,
        n_duplicate_timestamps=duplicates,
        n_missing_hours=n_missing,
        n_interpolated=n_missing - n_left,
        n_left_missing=n_left,
        first=first,
        last=last,
    )



def _interpolate_short_runs(series: pd.Series, max_hours: int) -> pd.Series:
    """Interpolate gaps of at most ``max_hours``, leaving longer ones untouched.

    ``Series.interpolate(limit=n)`` does something subtly different and worse:
    given a ten-hour outage it fills the first ``n`` hours and leaves the rest,
    so a gap that should have been rejected whole comes back partly invented
    and partly missing. Whether a gap is fillable is a property of the entire
    run, so runs are measured first and only short ones are filled.
    """
    missing = series.isna()
    if not missing.any():
        return series

    # Label each maximal run of NaNs, then keep only the short ones.
    run_id = (missing != missing.shift()).cumsum()
    run_length = missing.groupby(run_id).transform("sum")
    fillable = missing & (run_length <= max_hours)

    interpolated = series.interpolate(method="time", limit_area="inside")
    return series.where(~fillable, interpolated)


def load_series(
    path: str | Path, max_interpolate_hours: int = 3
) -> tuple[pd.Series, CleaningReport]:
    """Convenience: read and clean in one call."""
    return clean(load_raw(path), max_interpolate_hours)


def summarise(series: pd.Series) -> dict[str, float]:
    """Headline statistics, for the README and for drift reference."""
    return {
        "n_hours": int(len(series)),
        "n_missing": int(series.isna().sum()),
        "mean_mw": float(series.mean()),
        "min_mw": float(series.min()),
        "max_mw": float(series.max()),
        "std_mw": float(series.std()),
        "start": str(series.index.min()),
        "end": str(series.index.max()),
    }


def peak_hours(series: pd.Series) -> pd.Series:
    """Mean load by hour of day -- the daily shape the model has to learn."""
    return series.groupby(series.index.hour).mean()


def seasonal_strength(series: pd.Series, period: int) -> float:
    """Fraction of variance explained by the mean profile at a given period.

    A crude but honest measure of how much of the signal a purely seasonal
    model could capture, which is why the seasonal-naive baseline is hard to
    beat.
    """
    values = series.dropna()
    index = np.arange(len(values)) % period
    profile = pd.Series(values.to_numpy()).groupby(index).transform("mean")
    residual = values.to_numpy() - profile.to_numpy()
    return float(1.0 - np.var(residual) / np.var(values.to_numpy()))
