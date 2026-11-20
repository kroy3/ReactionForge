#!/usr/bin/env python3
"""
ReactionForge Training Script
==============================
Complete training workflow with checkpointing and evaluation.

Usage:
    python scripts/train.py --data_path data/reactions.csv --output_dir checkpoints/

The best model (by validation R²) is saved to ``<output_dir>/best_model.pt``
together with the constructor arguments needed to rebuild it:

    ckpt = torch.load('checkpoints/best_model.pt')
    model = ReactionForge(**ckpt['model_config'])
    model.load_state_dict(ckpt['model_state_dict'])
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from src.models.reactionforge import ReactionForge
from src.data.dataset import SuzukiMiyauraDataset
from src.training.trainer import ReactionForgeTrainer
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.utils.uncertainty import coverage, fit_uncertainty_scale


def parse_args():
    parser = argparse.ArgumentParser(description='Train ReactionForge')
    parser.add_argument('--data_path', type=str, required=True,
                        help='CSV of reactions (see docs/USER_GUIDE.md §6)')
    parser.add_argument('--output_dir', type=str, default='checkpoints/')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_wln_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--evidential_lambda', type=float, default=0.01,
                        help='Weight of the evidential regulariser')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early-stopping patience in epochs')
    parser.add_argument('--no_temporal_memory', action='store_true',
                        help='Do not feed reaction timestamps to the model')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    dataset = SuzukiMiyauraDataset(args.data_path)
    if len(dataset) < 3:
        sys.exit(f"Need at least 3 valid reactions to split, found {len(dataset)}.")

    # Feature sizes come from the data, so they always match the featuriser
    sample = dataset[0]
    node_feat_dim = sample['reactant'].x.shape[1]
    edge_feat_dim = sample['reactant'].edge_attr.shape[1]
    condition_dim = sample['conditions'].shape[0]

    # Split dataset (70 / 15 / 15), reproducible via --seed
    train_size = int(0.7 * len(dataset))
    val_size = max(1, int(0.15 * len(dataset)))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers)

    print(f"Split: {train_size} train, {val_size} val, {test_size} test")

    # Initialize model
    model_config = dict(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        condition_dim=condition_dim,
        hidden_dim=args.hidden_dim,
        num_wln_layers=args.num_wln_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    model = ReactionForge(**model_config)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize trainer
    optimizer = Adam(model.parameters(), lr=args.learning_rate,
                     weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    trainer = ReactionForgeTrainer(model, optimizer, device,
                                   lambda_reg=args.evidential_lambda,
                                   use_timestamps=not args.no_temporal_memory)

    # Callbacks
    best_path = os.path.join(args.output_dir, 'best_model.pt')
    early_stopping = EarlyStopping(patience=args.patience, mode='max')
    checkpoint = ModelCheckpoint(
        best_path,
        monitor='r2',
        mode='max',
        extra={'model_config': model_config,
               'use_temporal_memory': not args.no_temporal_memory}
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
    checkpoint_data = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint_data['model_state_dict'])

    # Calibrate uncertainty on the validation set: one factor that rescales the
    # predicted standard deviations to match the observed errors. The test set
    # is left untouched so its coverage is an honest check.
    y_val, p_val, var_val = trainer.collect_predictions(val_loader)
    scale = fit_uncertainty_scale(y_val, p_val, var_val)
    checkpoint_data['uncertainty_scale'] = scale
    torch.save(checkpoint_data, best_path)

    if test_size == 0:
        print("Test set is empty (dataset too small); skipping test evaluation.")
    else:
        test_metrics = trainer.evaluate(test_loader)
        print(f"Test R²: {test_metrics['r2']:.4f}")
        print(f"Test RMSE: {test_metrics['rmse']:.2f}%")
        print(f"Test MAE: {test_metrics['mae']:.2f}%")

        y_t, p_t, var_t = trainer.collect_predictions(test_loader)
        raw_std = var_t ** 0.5
        print(f"\nUncertainty calibration (scale {scale:.3f}, fitted on validation set)")
        print("Test reactions within ±1 / ±2 std (ideal ≈ 68% / 95%):")
        print(f"  before: {coverage(y_t, p_t, raw_std, 1):.0%} / {coverage(y_t, p_t, raw_std, 2):.0%}")
        print(f"  after:  {coverage(y_t, p_t, raw_std * scale, 1):.0%} / "
              f"{coverage(y_t, p_t, raw_std * scale, 2):.0%}")
    print("="*60)

    print(f"\n✅ Training complete! Best model saved to {best_path}")


if __name__ == '__main__':
    main()
