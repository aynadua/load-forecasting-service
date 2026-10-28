"""Input and prediction drift monitoring.

A forecasting service does not fail loudly. It keeps returning plausible
numbers while the world moves underneath it -- a new large customer connects,
a mild winter arrives, a meter changes calibration -- and nobody notices until
someone reconciles against actuals weeks later. Drift detection is the cheap
smoke alarm: it cannot tell you the model is wrong, only that the data no
longer resembles what the model was fitted on, which is a reason to look.

Two complementary tests:

* **Population Stability Index**, the standard in industry monitoring. The
  reference distribution is cut into quantile bins and the live distribution
  is compared bin by bin. The conventional reading is < 0.10 stable,
  0.10-0.25 moderate, > 0.25 significant.
* **Two-sample Kolmogorov-Smirnov**, which is more sensitive to a change in
  shape that leaves the bin proportions roughly intact.

Both are reported, because they disagree in informative ways.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25


def population_stability_index(
    reference: np.ndarray, live: np.ndarray, n_bins: int = 10, epsilon: float = 1e-6
) -> float:
    """PSI between a reference and a live sample."""
    reference = reference[np.isfinite(reference)]
    live = live[np.isfinite(live)]
    if len(reference) < n_bins or len(live) == 0:
        return float("nan")

    # Quantile edges from the reference, so each reference bin holds ~1/n_bins.
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(reference, quantiles))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    reference_share = np.histogram(reference, bins=edges)[0] / len(reference)
    live_share = np.histogram(live, bins=edges)[0] / len(live)

    reference_share = np.clip(reference_share, epsilon, None)
    live_share = np.clip(live_share, epsilon, None)
    return float(np.sum((live_share - reference_share) * np.log(live_share / reference_share)))


def classify(psi: float) -> str:
    """Conventional PSI bands."""
    if not np.isfinite(psi):
        return "unknown"
    if psi < PSI_MODERATE:
        return "stable"
    if psi < PSI_SIGNIFICANT:
        return "moderate"
    return "significant"


@dataclass
class DriftReport:
    """Per-feature drift, plus the worst offender."""

    features: dict[str, dict[str, Any]] = field(default_factory=dict)
    n_reference: int = 0
    n_live: int = 0

    @property
    def worst(self) -> tuple[str, float] | None:
        scored = [
            (name, entry["psi"])
            for name, entry in self.features.items()
            if np.isfinite(entry["psi"])
        ]
        return max(scored, key=lambda pair: pair[1]) if scored else None

    @property
    def status(self) -> str:
        worst = self.worst
        return classify(worst[1]) if worst else "unknown"

    def to_dict(self) -> dict[str, Any]:
        worst = self.worst
        return {
            "status": self.status,
            "worst_feature": worst[0] if worst else None,
            "worst_psi": worst[1] if worst else None,
            "n_reference": self.n_reference,
            "n_live": self.n_live,
            "features": self.features,
        }


class DriftMonitor:
    """Holds a reference sample and scores live batches against it."""

    def __init__(self, reference: pd.DataFrame, n_bins: int = 10) -> None:
        if reference.empty:
            raise ValueError("reference frame is empty")
        self.reference = reference
        self.n_bins = n_bins

    def score(self, live: pd.DataFrame) -> DriftReport:
        """Compare a live batch against the reference, feature by feature."""
        shared = [c for c in self.reference.columns if c in live.columns]
        report = DriftReport(n_reference=len(self.reference), n_live=len(live))

        for name in shared:
            reference_values = self.reference[name].to_numpy(dtype=float)
            live_values = live[name].to_numpy(dtype=float)
            psi = population_stability_index(reference_values, live_values, self.n_bins)

            entry: dict[str, Any] = {"psi": psi, "status": classify(psi)}
            if len(live_values) >= 20:
                statistic, p_value = ks_2samp(reference_values, live_values)
                entry["ks_statistic"] = float(statistic)
                entry["ks_p_value"] = float(p_value)
            report.features[name] = entry

        return report

    @classmethod
    def from_training(cls, x: pd.DataFrame, sample: int = 20000, seed: int = 0) -> DriftMonitor:
        """Build a monitor from a (subsampled) training design matrix."""
        if len(x) > sample:
            x = x.sample(sample, random_state=seed)
        return cls(reference=x)
