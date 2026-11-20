"""Unit tests for data processing"""

import pytest
from src.data.featurization import atom_features, bond_features, smiles_to_features
from rdkit import Chem


def test_atom_features():
    """Test atom featurization"""
    mol = Chem.MolFromSmiles('CCO')
    atom = mol.GetAtomWithIdx(0)  # Carbon
    
    features = atom_features(atom)
    
    # 118 element + 6 degree + 5 hybridisation + 1 aromatic + 5 H count
    # + 5 formal charge + 3 chirality + 1 in-ring
    assert len(features) == 144
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
    assert atom_feat.shape[1] == 144  # Feature dimension
    assert edge_index.shape[0] == 2  # Edge connectivity
    assert edge_feat.shape[1] == 6  # Edge feature dimension


def test_dataset_graph_features():
    """The featuriser used for training (src/data/dataset.py) adds explicit
    hydrogens and produces 150 atom features and 6 bond features."""
    from src.data.dataset import smiles_to_graph

    graph = smiles_to_graph('c1ccc(Br)cc1')  # C6H5Br -> 12 atoms, 12 bonds

    assert graph.x.shape == (12, 150)
    assert graph.edge_index.shape == (2, 24)  # both directions
    assert graph.edge_attr.shape == (24, 6)
    assert smiles_to_graph('notasmiles') is None


if __name__ == '__main__':
    pytest.main([__file__])
