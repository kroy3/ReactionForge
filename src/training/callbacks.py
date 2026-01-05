"""
Training Callbacks
==================
Early stopping, checkpointing, and logging.
"""

import torch
import numpy as np


class EarlyStopping:
    """Stop training when validation metric stops improving"""
    
    def __init__(self, patience=20, min_delta=0.0001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class ModelCheckpoint:
    """Save best model based on validation metric"""
    
    def __init__(self, filepath, monitor='val_r2', mode='max'):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        
    def __call__(self, model, metrics):
        score = metrics.get(self.monitor, 0)
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, metrics)
            return True
        
        if self.mode == 'max':
            improved = score > self.best_score
        else:
            improved = score < self.best_score
        
        if improved:
            self.best_score = score
            self.save_checkpoint(model, metrics)
            return True
        
        return False
    
    def save_checkpoint(self, model, metrics):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'best_score': self.best_score
        }, self.filepath)
