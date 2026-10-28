"""Baselines the model has to beat.

Electricity load is overwhelmingly periodic, so these are not weak. Seasonal
naive at a one-week lag reuses the load from the same hour of the same weekday
and is, for many hours of the year, difficult to improve on. Any claim that a
learned model is useful has to be made against this, not against a mean.

**The history a baseline is allowed to see.** A seasonal naive forecast for
tomorrow at 18:00 reads the load recorded last Tuesday at 18:00. In operation
that value is known -- it is a week old. So the baseline is entitled to the
actual series up to ``horizon`` hours before each target, exactly like the
model's ``lag_168h`` feature, and restricting it to the training split instead
would be a rigged comparison rather than a conservative one.

The guard that keeps this honest is :meth:`SeasonalNaive.validate_for_horizon`:
a baseline whose period is shorter than the forecast horizon would be reading
the future, and refuses to run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SeasonalNaive:
    """Predict the observed value one seasonal period earlier."""

    def __init__(self, period_hours: int = 168, name: str | None = None) -> None:
        self.period_hours = period_hours
        self.name = name or f"seasonal_naive_{period_hours}h"
        self._history: pd.Series | None = None

    def validate_for_horizon(self, horizon: int) -> None:
        """Refuse to use a lag the forecaster could not have observed."""
        if self.period_hours < horizon:
            raise ValueError(
                f"{self.name} reads {self.period_hours}h back, which is inside the "
                f"{horizon}h forecast horizon; that value is not observable at issue time"
            )

    def fit(self, history: pd.Series) -> SeasonalNaive:
        """Store the observed series the forecast reads back into."""
        self._history = history.sort_index()
        return self

    def predict(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self._history is None:
            raise RuntimeError("call fit before predict")
        wanted = pd.DatetimeIndex(index) - pd.Timedelta(hours=self.period_hours)
        # copy=True: reindex can hand back a read-only view, and the
        # fallback below writes into it.
        values = self._history.reindex(wanted).to_numpy(dtype=float, copy=True)

        missing = np.isnan(values)
        if missing.any():
            # Only reachable at the very start of a series, before one full
            # period of history exists.
            values[missing] = float(self._history.mean())
        return values


def build(name: str) -> SeasonalNaive:
    """Look up a baseline by name."""
    if name == "seasonal_naive_weekly":
        return SeasonalNaive(period_hours=168, name="seasonal naive (1 week)")
    if name == "seasonal_naive_daily":
        return SeasonalNaive(period_hours=24, name="seasonal naive (1 day)")
    raise ValueError(f"unknown baseline {name!r}")


BASELINES: tuple[str, ...] = ("seasonal_naive_weekly", "seasonal_naive_daily")
