"""
Visualization Utilities
========================
Plotting functions for results and analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_predictions(y_true, y_pred, uncertainty=None, save_path=None):
    """Plot predicted vs actual yields"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if uncertainty is not None:
        scatter = ax.scatter(y_true, y_pred, c=uncertainty, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Uncertainty')
    else:
        ax.scatter(y_true, y_pred, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    ax.set_xlabel('True Yield (%)')
    ax.set_ylabel('Predicted Yield (%)')
    ax.set_title('Predicted vs True Yields')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_learning_curves(history, save_path=None):
    """Plot training and validation metrics over epochs"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    
    # R² curves
    axes[1].plot(history['train_r2'], label='Train R²')
    axes[1].plot(history['val_r2'], label='Val R²')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('R² Score')
    axes[1].set_title('Model Performance')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_attention_weights(attention, molecule, save_path=None):
    """Visualize attention weights on molecular structure"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap of attention matrix
    sns.heatmap(attention, cmap='YlOrRd', ax=ax)
    ax.set_title('Cross-Attention Weights')
    ax.set_xlabel('Product Atoms')
    ax.set_ylabel('Reactant Atoms')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
