"""
Uncertainty Calibration
========================
Tools for uncertainty quantification and calibration.
"""

import numpy as np


def fit_uncertainty_scale(y_true, y_pred, variance):
    """
    Fit one factor ``s`` so that the predictive distribution N(y_pred, s^2 * variance)
    best matches the observed errors (closed-form Gaussian maximum likelihood).

    Fit it on a validation set, never on the data you report results on.

    Args:
        y_true, y_pred: true and predicted values (same scale)
        variance: predicted *variance* on that scale, e.g. the model's
            ``epistemic_uncertainty + aleatoric_uncertainty``

    Returns:
        float: multiply standard deviations by ``s`` (variances by ``s**2``).
        s > 1 means the model was overconfident, s < 1 underconfident.
    """
    y_true, y_pred, variance = (np.asarray(a, dtype=float).ravel()
                                for a in (y_true, y_pred, variance))
    if len(y_true) == 0:
        raise ValueError("Need at least one prediction to fit an uncertainty scale")
    variance = np.maximum(variance, 1e-12)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2 / variance)))


def calibrate_uncertainty(y_true, y_pred, epistemic, aleatoric):
    """
    Rescale epistemic and aleatoric *variances* so that their sum matches the
    observed errors (temperature scaling; see ``fit_uncertainty_scale``).

    Args:
        y_true: True values
        y_pred: Predicted values
        epistemic: Epistemic variance (model output ``epistemic_uncertainty``)
        aleatoric: Aleatoric variance (model output ``aleatoric_uncertainty``)

    Returns:
        tuple: Calibrated (epistemic, aleatoric) variances
    """
    epistemic = np.asarray(epistemic, dtype=float)
    aleatoric = np.asarray(aleatoric, dtype=float)
    scale = fit_uncertainty_scale(y_true, y_pred, epistemic + aleatoric)
    return epistemic * scale ** 2, aleatoric * scale ** 2


def coverage(y_true, y_pred, std, z=1.0):
    """Fraction of true values within ``z`` standard deviations of the prediction.
    For calibrated Gaussian uncertainty this is about 68% (z=1) and 95% (z=2)."""
    y_true, y_pred, std = (np.asarray(a, dtype=float).ravel() for a in (y_true, y_pred, std))
    return float(np.mean(np.abs(y_true - y_pred) <= z * std))


def uncertainty_based_sampling(epistemic, n_samples):
    """
    Select samples with highest epistemic uncertainty for active learning.
    
    Args:
        epistemic: Epistemic uncertainty values
        n_samples: Number of samples to select
        
    Returns:
        np.array: Indices of selected samples
    """
    return np.argsort(epistemic)[-n_samples:]


def reliability_diagram(y_true, y_pred, uncertainty, n_bins=10):
    """
    Create data for reliability diagram.
    
    Returns:
        dict: Bin statistics for plotting
    """
    indices = np.argsort(uncertainty)
    y_true = y_true[indices]
    y_pred = y_pred[indices]
    uncertainty = uncertainty[indices]
    
    bin_boundaries = np.linspace(0, len(y_true), n_bins + 1).astype(int)
    
    bin_confidences = []
    bin_accuracies = []
    
    for i in range(n_bins):
        start, end = bin_boundaries[i], bin_boundaries[i + 1]
        if start == end:
            continue
        
        bin_unc = uncertainty[start:end]
        bin_errors = np.abs(y_true[start:end] - y_pred[start:end])
        
        # Average confidence (1 - normalized uncertainty)
        confidence = 1 - np.mean(bin_unc / uncertainty.max())
        
        # Average accuracy (1 - normalized error)
        accuracy = 1 - np.mean(bin_errors / np.abs(y_true).max())
        
        bin_confidences.append(confidence)
        bin_accuracies.append(accuracy)
    
    return {
        'confidences': np.array(bin_confidences),
        'accuracies': np.array(bin_accuracies)
    }
