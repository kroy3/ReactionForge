#!/usr/bin/env python3
"""
Evaluation script for trained ReactionForge models
==================================================
Score a model trained by scripts/train.py on a labelled CSV (one with a
``yield`` column).

Usage:
    python scripts/evaluate.py \\
        --checkpoint checkpoints/best_model.pt \\
        --data_path held_out.csv \\
        [--output_csv per_reaction.csv] [--plot parity.png]

Evaluate on reactions the model was NOT trained on. Running it on the
training CSV reports how well the model memorised that data, not how well
it generalises.
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from src.inference import load_model, predict_dataframe
from src.training.metrics import calculate_metrics, expected_calibration_error


def coverage(y_true, y_pred, std, z):
    """Fraction of reactions whose true yield lies within z standard deviations."""
    return float(np.mean(np.abs(y_true - y_pred) <= z * std))


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained ReactionForge model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='best_model.pt written by scripts/train.py')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Labelled CSV (must contain a yield column)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Optional: write per-reaction predictions and errors here')
    parser.add_argument('--plot', type=str, default=None,
                        help='Optional: save a predicted-vs-true plot (e.g. parity.png)')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    try:
        model, use_timestamps = load_model(args.checkpoint, device)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"Error: {e}")

    # Load data
    df = pd.read_csv(args.data_path)
    if 'yield' not in df.columns:
        sys.exit("Error: evaluation needs a 'yield' column. "
                 "To predict unlabelled reactions use scripts/predict.py.")

    # Evaluate
    print(f"Evaluating model on {len(df)} reactions...")
    try:
        preds = predict_dataframe(model, df, use_timestamps=use_timestamps,
                                  batch_size=args.batch_size, device=device)
    except ValueError as e:
        sys.exit(f"Error: {e}")

    scored = (preds['status'] == 'ok') & df['yield'].notna()
    skipped = len(df) - int(scored.sum())
    if scored.sum() < 2:
        sys.exit("Error: fewer than 2 reactions could be scored.")

    y_true = df.loc[scored, 'yield'].to_numpy(dtype=float)
    y_pred = preds.loc[scored, 'predicted_yield'].to_numpy(dtype=float)
    std = preds.loc[scored, 'uncertainty_std'].to_numpy(dtype=float)

    m = calculate_metrics(y_true, y_pred)
    ece = expected_calibration_error(y_true / 100, y_pred / 100, std / 100)
    cov1, cov2 = coverage(y_true, y_pred, std, 1), coverage(y_true, y_pred, std, 2)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Reactions scored: {int(scored.sum())}"
          + (f"  (skipped {skipped}: invalid SMILES or missing yield)" if skipped else ""))
    print(f"R²:   {m['r2']:.4f}")
    print(f"RMSE: {m['rmse']:.2f}%")
    print(f"MAE:  {m['mae']:.2f}%")
    print(f"ECE:  {ece:.4f}")
    print("Uncertainty coverage (ideal for a calibrated Gaussian: 68% / 95%):")
    print(f"  true yield within ±1 std: {cov1:.0%}")
    print(f"  true yield within ±2 std: {cov2:.0%}")
    print("=" * 60)

    if args.output_csv:
        result = pd.concat([df, preds], axis=1)
        result['abs_error'] = (result['predicted_yield'] - result['yield']).abs()
        out_dir = os.path.dirname(args.output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        result.to_csv(args.output_csv, index=False)
        print(f"Per-reaction results -> {args.output_csv}")

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        from src.utils.visualization import plot_predictions
        plot_predictions(y_true, y_pred, uncertainty=std, save_path=args.plot)
        print(f"Parity plot -> {args.plot}")


if __name__ == '__main__':
    main()
