#!/usr/bin/env python3
"""Batch prediction script"""

import argparse
import torch
import pandas as pd

from src.models.reactionforge import ReactionForge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    args = parser.parse_args()
    
    # Load model
    checkpoint = torch.load(args.checkpoint)
    model = ReactionForge()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load reactions
    df = pd.read_csv(args.input_csv)
    
    # Make predictions
    print(f"Making predictions for {len(df)} reactions...")
    # Implementation continues...


if __name__ == '__main__':
    main()
