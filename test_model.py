"""Unit tests for ReactionForge model"""

import pytest
import torch
from src.models.reactionforge import ReactionForge


def test_model_forward():
    """Test forward pass with dummy data"""
    model = ReactionForge(node_feat_dim=154, edge_feat_dim=6, condition_dim=10)
    
    # Create dummy batch
    batch = {
        'reactant_x': torch.randn(10, 154),
        'reactant_edge_index': torch.randint(0, 10, (2, 20)),
        'reactant_edge_attr': torch.randn(20, 6),
        'product_x': torch.randn(12, 154),
        'product_edge_index': torch.randint(0, 12, (2, 24)),
        'product_edge_attr': torch.randn(24, 6),
        'conditions': torch.randn(1, 10),
        'yield': torch.tensor([0.85])
    }
    
    output = model(batch)
    
    assert 'yield_mean' in output
    assert 'uncertainty' in output
    assert output['yield_mean'].shape == (1,)


def test_model_parameters():
    """Test that model has expected number of parameters"""
    model = ReactionForge(hidden_dim=128)
    num_params = sum(p.numel() for p in model.parameters())
    
    # Should be around 3.2M parameters
    assert 3_000_000 < num_params < 4_000_000


if __name__ == '__main__':
    pytest.main([__file__])
