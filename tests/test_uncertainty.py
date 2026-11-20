"""Tests for uncertainty calibration utilities"""

import numpy as np
import pytest

from src.utils.uncertainty import calibrate_uncertainty, coverage, fit_uncertainty_scale


def overconfident_predictions(true_scale=2.5, n=20000, seed=0):
    """Errors drawn with std = true_scale * the model's claimed std."""
    rng = np.random.default_rng(seed)
    variance = rng.uniform(0.001, 0.01, n)  # model's claimed variances
    y_pred = rng.uniform(0, 1, n)
    y_true = y_pred + rng.normal(0, true_scale * np.sqrt(variance))
    return y_true, y_pred, variance


@pytest.mark.parametrize('true_scale', [0.4, 1.0, 2.5])
def test_fit_uncertainty_scale_recovers_known_scale(true_scale):
    y_true, y_pred, variance = overconfident_predictions(true_scale)
    assert fit_uncertainty_scale(y_true, y_pred, variance) == pytest.approx(true_scale, rel=0.03)


def test_calibration_restores_gaussian_coverage():
    y_true, y_pred, variance = overconfident_predictions(true_scale=2.5)
    raw_std = np.sqrt(variance)
    assert coverage(y_true, y_pred, raw_std, 1) < 0.4  # badly overconfident

    s = fit_uncertainty_scale(y_true, y_pred, variance)
    assert coverage(y_true, y_pred, raw_std * s, 1) == pytest.approx(0.683, abs=0.02)
    assert coverage(y_true, y_pred, raw_std * s, 2) == pytest.approx(0.954, abs=0.02)


def test_calibrate_uncertainty_treats_inputs_as_variances():
    y_true, y_pred, variance = overconfident_predictions(true_scale=2.0)
    epistemic, aleatoric = 0.25 * variance, 0.75 * variance

    epi_cal, ale_cal = calibrate_uncertainty(y_true, y_pred, epistemic, aleatoric)

    # Variances scale by s^2 (~4); the epistemic/aleatoric split is preserved
    assert np.mean((epi_cal + ale_cal) / variance) == pytest.approx(4.0, rel=0.05)
    assert np.allclose(epi_cal / (epi_cal + ale_cal), 0.25)


def test_fit_uncertainty_scale_rejects_empty_input():
    with pytest.raises(ValueError):
        fit_uncertainty_scale([], [], [])
