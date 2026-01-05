#!/usr/bin/env python3
"""
ReactionForge Training Script
==============================
Complete training workflow with checkpointing and evaluation.

Usage:
    python scripts/train.py --data_path data/reactions.csv --output_dir checkpoints/
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from src.models.reactionforge import ReactionForge
from src.data.dataset import SuzukiMiyauraDataset
from src.training.trainer import ReactionForgeTrainer
from src.training.callbacks import EarlyStopping, ModelCheckpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Train ReactionForge')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='checkpoints/')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    dataset = SuzukiMiyauraDataset(args.data_path)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    print(f"Split: {train_size} train, {val_size} val, {test_size} test")
    
    # Initialize model
    model = ReactionForge(
        node_feat_dim=154,
        edge_feat_dim=6,
        condition_dim=10,
        hidden_dim=args.hidden_dim
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    trainer = ReactionForgeTrainer(model, optimizer, device)
    
    # Callbacks
    early_stopping = EarlyStopping(patience=20, mode='max')
    checkpoint = ModelCheckpoint(
        os.path.join(args.output_dir, 'best_model.pt'),
        monitor='r2',
        mode='max'
    )
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader)
        
        # Evaluate
        val_metrics = trainer.evaluate(val_loader)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val R²: {val_metrics['r2']:.4f}, "
              f"RMSE: {val_metrics['rmse']:.2f}%, "
              f"MAE: {val_metrics['mae']:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_metrics['r2'])
        
        # Checkpointing
        if checkpoint(model, val_metrics):
            print(f"✓ Saved checkpoint (R²={val_metrics['r2']:.4f})")
        
        # Early stopping
        if early_stopping(val_metrics['r2']):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    
    # Load best model
    checkpoint_data = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint_data['model_state_dict'])
    
    test_metrics = trainer.evaluate(test_loader)
    print(f"Test R²: {test_metrics['r2']:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.2f}%")
    print(f"Test MAE: {test_metrics['mae']:.2f}%")
    print("="*60)
    
    print(f"\n✅ Training complete!")


if __name__ == '__main__':
    main()
