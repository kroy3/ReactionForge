"""Unit tests for training pipeline"""

import pytest
import torch
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.metrics import calculate_metrics


def test_early_stopping():
    """Test early stopping callback"""
    early_stop = EarlyStopping(patience=3, mode='max')
    
    assert not early_stop(0.90)
    assert not early_stop(0.92)
    assert not early_stop(0.91)  # No improvement
    assert not early_stop(0.91)  # Still no improvement
    assert not early_stop(0.91)  # Third time
    assert early_stop.early_stop  # Should trigger


def test_metrics_calculation():
    """Test metric computation"""
    y_true = torch.tensor([0.8, 0.7, 0.9, 0.85])
    y_pred = torch.tensor([0.82, 0.68, 0.88, 0.87])
    
    metrics = calculate_metrics(y_true.numpy(), y_pred.numpy())
    
    assert 'r2' in metrics
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 0 < metrics['r2'] < 1


if __name__ == '__main__':
    pytest.main([__file__])
