"""
Evaluation Metrics
==================
Metrics for regression and uncertainty quantification.
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)
    }


def expected_calibration_error(y_true, y_pred, uncertainty, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).
    
    Measures how well predicted uncertainties match actual errors.
    Lower is better (perfect calibration = 0).
    """
    # Sort by uncertainty
    indices = np.argsort(uncertainty)
    y_true = y_true[indices]
    y_pred = y_pred[indices]
    uncertainty = uncertainty[indices]
    
    # Create bins
    bin_boundaries = np.linspace(0, len(y_true), n_bins + 1).astype(int)
    
    ece = 0.0
    for i in range(n_bins):
        start, end = bin_boundaries[i], bin_boundaries[i + 1]
        if start == end:
            continue
        
        bin_true = y_true[start:end]
        bin_pred = y_pred[start:end]
        bin_unc = uncertainty[start:end]
        
        # Predicted confidence (inverse of uncertainty)
        pred_conf = 1 - np.mean(bin_unc)
        
        # Actual accuracy
        actual_acc = 1 - np.mean(np.abs(bin_true - bin_pred))
        
        # Weighted difference
        ece += (end - start) / len(y_true) * np.abs(pred_conf - actual_acc)
    
    return ece


def uncertainty_decomposition(epistemic, aleatoric):
    """
    Decompose total uncertainty into epistemic and aleatoric components.
    
    Returns:
        dict: Statistics about uncertainty decomposition
    """
    total = np.sqrt(epistemic**2 + aleatoric**2)
    
    return {
        'epistemic_mean': np.mean(epistemic),
        'aleatoric_mean': np.mean(aleatoric),
        'total_mean': np.mean(total),
        'epistemic_ratio': np.mean(epistemic / (total + 1e-8))
    }
