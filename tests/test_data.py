"""Unit tests for data processing"""

import pytest
from src.data.featurization import atom_features, bond_features, smiles_to_features
from rdkit import Chem


def test_atom_features():
    """Test atom featurization"""
    mol = Chem.MolFromSmiles('CCO')
    atom = mol.GetAtomWithIdx(0)  # Carbon
    
    features = atom_features(atom)
    
    assert len(features) == 154
    assert features.sum() > 0  # Should have some non-zero features


def test_bond_features():
    """Test bond featurization"""
    mol = Chem.MolFromSmiles('C=C')
    bond = mol.GetBondWithIdx(0)  # Double bond
    
    features = bond_features(bond)
    
    assert len(features) == 6
    assert features[1] == 1  # Should be marked as double bond


def test_smiles_to_features():
    """Test full SMILES to features conversion"""
    smiles = 'c1ccccc1'  # Benzene
    
    atom_feat, edge_index, edge_feat = smiles_to_features(smiles)
    
    assert atom_feat.shape[0] == 6  # 6 atoms
    assert atom_feat.shape[1] == 154  # Feature dimension
    assert edge_index.shape[0] == 2  # Edge connectivity
    assert edge_feat.shape[1] == 6  # Edge feature dimension


if __name__ == '__main__':
    pytest.main([__file__])
