"""
Uncertainty Calibration
========================
Tools for uncertainty quantification and calibration.
"""

import numpy as np
from scipy.stats import norm


def calibrate_uncertainty(y_true, y_pred, epistemic, aleatoric):
    """
    Calibrate uncertainty estimates using temperature scaling.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        epistemic: Epistemic uncertainty
        aleatoric: Aleatoric uncertainty
        
    Returns:
        tuple: Calibrated (epistemic, aleatoric)
    """
    # Calculate residuals
    residuals = np.abs(y_true - y_pred)
    
    # Find optimal temperature parameter
    from scipy.optimize import minimize
    
    def objective(temp):
        total_unc = np.sqrt(epistemic**2 + aleatoric**2) * temp
        log_likelihood = -np.sum(norm.logpdf(residuals, scale=total_unc))
        return log_likelihood
    
    result = minimize(objective, x0=1.0, bounds=[(0.1, 10.0)])
    optimal_temp = result.x[0]
    
    # Apply temperature scaling
    epistemic_cal = epistemic * optimal_temp
    aleatoric_cal = aleatoric * optimal_temp
    
    return epistemic_cal, aleatoric_cal


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
