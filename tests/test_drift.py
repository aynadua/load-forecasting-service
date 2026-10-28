"""Drift detection calibration."""

import numpy as np
import pandas as pd
import pytest

from loadfc.drift import (
    PSI_SIGNIFICANT,
    DriftMonitor,
    classify,
    population_stability_index,
)


@pytest.fixture
def reference():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.normal(0, 1, 5000), "b": rng.normal(10, 2, 5000)})


def test_identical_distributions_give_near_zero_psi():
    rng = np.random.default_rng(1)
    sample = rng.normal(0, 1, 5000)
    other = rng.normal(0, 1, 5000)
    assert population_stability_index(sample, other) < 0.05


def test_a_shifted_distribution_is_flagged_significant():
    rng = np.random.default_rng(2)
    reference_values = rng.normal(0, 1, 5000)
    shifted = rng.normal(2.0, 1, 5000)
    assert population_stability_index(reference_values, shifted) > PSI_SIGNIFICANT


def test_psi_grows_monotonically_with_the_size_of_the_shift():
    rng = np.random.default_rng(3)
    reference_values = rng.normal(0, 1, 20000)
    scores = [
        population_stability_index(reference_values, rng.normal(shift, 1, 20000))
        for shift in (0.0, 0.25, 0.5, 1.0, 2.0)
    ]
    assert scores == sorted(scores)


def test_classification_bands():
    assert classify(0.01) == "stable"
    assert classify(0.15) == "moderate"
    assert classify(0.9) == "significant"
    assert classify(float("nan")) == "unknown"


def test_psi_handles_a_constant_feature():
    constant = np.ones(1000)
    assert population_stability_index(constant, constant) == 0.0


def test_monitor_reports_the_worst_feature(reference):
    rng = np.random.default_rng(4)
    live = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(16, 2, 2000)})

    report = DriftMonitor(reference).score(live)
    assert report.worst[0] == "b"
    assert report.status == "significant"
    assert report.to_dict()["n_live"] == 2000


def test_monitor_ignores_columns_the_reference_lacks(reference):
    rng = np.random.default_rng(5)
    live = pd.DataFrame({"a": rng.normal(0, 1, 500), "unseen": rng.normal(0, 1, 500)})
    report = DriftMonitor(reference).score(live)
    assert set(report.features) == {"a"}


def test_monitor_rejects_an_empty_reference():
    with pytest.raises(ValueError, match="empty"):
        DriftMonitor(pd.DataFrame())


def test_from_training_subsamples(reference):
    monitor = DriftMonitor.from_training(reference, sample=100)
    assert len(monitor.reference) == 100
