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


def test_trainer_end_to_end(tmp_path):
    """Trainer runs a real dataset batch through the model (regression test
    for scripts/train.py crashing on the first batch)."""
    from torch_geometric.loader import DataLoader
    from src.data.dataset import SuzukiMiyauraDataset, generate_synthetic_dataset
    from src.models.reactionforge import ReactionForge
    from src.training.trainer import ReactionForgeTrainer

    csv = tmp_path / 'reactions.csv'
    generate_synthetic_dataset(num_samples=24, output_path=str(csv))
    dataset = SuzukiMiyauraDataset(str(csv))
    loader = DataLoader(dataset, batch_size=8)

    node_dim = dataset[0]['reactant'].x.shape[1]
    model = ReactionForge(node_feat_dim=node_dim, edge_feat_dim=6, condition_dim=10,
                          hidden_dim=32, num_heads=4)
    trainer = ReactionForgeTrainer(model, torch.optim.Adam(model.parameters()), 'cpu')

    loss = trainer.train_epoch(loader)
    metrics = trainer.evaluate(loader)

    assert torch.isfinite(torch.tensor(loss))
    assert set(metrics) == {'loss', 'r2', 'rmse', 'mae'}
    assert metrics['rmse'] >= 0


def test_dataset_skips_invalid_smiles(tmp_path):
    """Rows RDKit cannot parse are dropped instead of crashing batching."""
    from src.data.dataset import SuzukiMiyauraDataset

    csv = tmp_path / 'reactions.csv'
    csv.write_text('reactant_smiles,product_smiles,yield\n'
                   'c1ccc(Br)cc1,c1ccc(-c2ccccc2)cc1,80\n'
                   'notasmiles,c1ccccc1,50\n')
    assert len(SuzukiMiyauraDataset(str(csv))) == 1


if __name__ == '__main__':
    pytest.main([__file__])
