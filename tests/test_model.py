"""Unit tests for ReactionForge model"""

import pytest
import torch
from torch_geometric.data import Batch

from src.data.dataset import smiles_to_graph
from src.models.reactionforge import ReactionForge, evidential_loss


def make_batch(smiles_list):
    return Batch.from_data_list([smiles_to_graph(s) for s in smiles_list])


@pytest.fixture
def inputs():
    reactants = make_batch(['c1ccc(Br)cc1', 'c1ccc(I)cc1'])
    products = make_batch(['c1ccc(-c2ccccc2)cc1', 'c1ccc(-c2ccccc2)cc1'])
    conditions = torch.rand(2, 10)
    return reactants, products, conditions


def test_model_forward(inputs):
    """Forward pass on real molecular graphs with one-hot bond features."""
    model = ReactionForge(node_feat_dim=150, edge_feat_dim=6, condition_dim=10).eval()

    with torch.no_grad():
        output = model(*inputs, timestamps=torch.tensor([[0], [1]]))

    assert output['yield_mean'].shape == (2, 1)
    assert output['selectivity_logits'].shape == (2, 3)
    for key in ('uncertainty', 'epistemic_uncertainty', 'aleatoric_uncertainty'):
        assert output[key].shape == (2, 1)
        assert (output[key] > 0).all()
    assert torch.allclose(output['uncertainty'],
                          output['epistemic_uncertainty'] + output['aleatoric_uncertainty'])
    assert output['temporal_state'] is not None


def test_model_forward_without_timestamps(inputs):
    model = ReactionForge(node_feat_dim=150, edge_feat_dim=6, condition_dim=10).eval()
    with torch.no_grad():
        output = model(*inputs)
    assert output['yield_mean'].shape == (2, 1)
    assert output['temporal_state'] is None


def test_evidential_loss_backpropagates(inputs):
    model = ReactionForge(node_feat_dim=150, edge_feat_dim=6, condition_dim=10)
    gamma, nu, alpha, beta = model(*inputs)['yield_params']

    loss = evidential_loss(gamma, nu, alpha, beta, torch.tensor([[0.8], [0.6]]))
    loss.backward()

    assert torch.isfinite(loss)
    assert model.atom_encoder[0].weight.grad is not None


def test_model_parameters():
    """Size of the model trained by scripts/train.py with default settings.
    docs/USER_GUIDE.md quotes this figure (~1.03 M); update both together."""
    model = ReactionForge(node_feat_dim=150, edge_feat_dim=6, condition_dim=10,
                          hidden_dim=128)
    num_params = sum(p.numel() for p in model.parameters())

    assert 1_000_000 < num_params < 1_100_000


def test_hidden_dim_must_divide_heads():
    with pytest.raises(AssertionError):
        ReactionForge(hidden_dim=100, num_heads=8)


if __name__ == '__main__':
    pytest.main([__file__])
