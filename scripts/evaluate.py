#!/usr/bin/env python3
"""Evaluation script for trained ReactionForge models"""

import argparse
import torch
from torch_geometric.loader import DataLoader

from src.models.reactionforge import ReactionForge
from src.data.dataset import SuzukiMiyauraDataset
from src.training.metrics import calculate_metrics, expected_calibration_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()
    
    # Load model
    checkpoint = torch.load(args.checkpoint)
    model = ReactionForge(node_feat_dim=154, edge_feat_dim=6, condition_dim=10)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data
    dataset = SuzukiMiyauraDataset(args.data_path)
    loader = DataLoader(dataset, batch_size=64)
    
    # Evaluate
    print("Evaluating model...")
    # Implementation continues...
    

if __name__ == '__main__':
    main()
