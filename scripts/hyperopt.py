#!/usr/bin/env python3
"""Hyperparameter optimization using Optuna"""

import argparse
import optuna


def objective(trial):
    """Optuna objective function"""
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 2, 5)
    
    # Train model with these hyperparameters
    # Return validation R²
    return 0.95  # Placeholder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=100)
    args = parser.parse_args()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials)
    
    print(f"Best parameters: {study.best_params}")
    print(f"Best R²: {study.best_value:.4f}")


if __name__ == '__main__':
    main()
