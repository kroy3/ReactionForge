#!/usr/bin/env python3
"""
Batch prediction script
=======================
Predict yields for every reaction in a CSV with a model trained by
scripts/train.py.

Usage:
    python scripts/predict.py \\
        --checkpoint checkpoints/best_model.pt \\
        --input_csv candidates.csv \\
        --output_csv predictions.csv

The output is the input CSV with these columns appended:
    predicted_yield   yield in percent
    uncertainty_std   predictive standard deviation, percentage points
    epistemic_var     model uncertainty (variance, 0-1 scale)
    aleatoric_var     data noise (variance, 0-1 scale)
    status            'ok', or why the row could not be predicted
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch

from src.inference import PREDICTION_COLUMNS, load_model, predict_dataframe


def main():
    parser = argparse.ArgumentParser(description='Predict reaction yields')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='best_model.pt written by scripts/train.py')
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Reactions to predict (reactant_smiles, product_smiles, conditions)')
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model, use_timestamps = load_model(args.checkpoint, device)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"Error: {e}")

    df = pd.read_csv(args.input_csv)
    clash = [c for c in PREDICTION_COLUMNS if c in df.columns]
    if clash:
        sys.exit(f"Input already has output column(s) {clash}; rename or drop them first.")

    print(f"Making predictions for {len(df)} reactions...")
    try:
        preds = predict_dataframe(model, df, use_timestamps=use_timestamps,
                                  batch_size=args.batch_size, device=device)
    except ValueError as e:
        sys.exit(f"Error: {e}")

    result = pd.concat([df, preds], axis=1)
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result.to_csv(args.output_csv, index=False)

    n_ok = int((preds['status'] == 'ok').sum())
    print(f"Predicted {n_ok}/{len(df)} reactions -> {args.output_csv}")
    if n_ok < len(df):
        print(f"⚠️  {len(df) - n_ok} rows could not be predicted; see the 'status' column.")


if __name__ == '__main__':
    main()
