"""
Molecular Featurization
========================
Convert molecules to numerical features for neural networks.
"""

import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np


def atom_features(atom):
    """
    Generate feature vector for an atom.
    
    Returns 154-dimensional feature vector including:
    - One-hot encoded atom type (118 elements)
    - Degree (one-hot, 0-5)
    - Hybridization (one-hot, SP/SP2/SP3/SP3D/SP3D2)
    - Aromaticity (boolean)
    - Number of hydrogens (one-hot, 0-4)
    - Formal charge (-2 to +2, one-hot)
    - Chirality (R/S/unspecified)
    - In ring (boolean)
    """
    features = []
    
    # Atomic number (one-hot encoding for first 118 elements)
    atom_type = [0] * 118
    if atom.GetAtomicNum() < 118:
        atom_type[atom.GetAtomicNum()] = 1
    features.extend(atom_type)
    
    # Degree (0-5)
    degree = [0] * 6
    if atom.GetDegree() < 6:
        degree[atom.GetDegree()] = 1
    features.extend(degree)
    
    # Hybridization
    hyb_types = [Chem.rdchem.HybridizationType.SP,
                 Chem.rdchem.HybridizationType.SP2,
                 Chem.rdchem.HybridizationType.SP3,
                 Chem.rdchem.HybridizationType.SP3D,
                 Chem.rdchem.HybridizationType.SP3D2]
    hyb = [0] * 5
    try:
        hyb[hyb_types.index(atom.GetHybridization())] = 1
    except:
        pass
    features.extend(hyb)
    
    # Aromatic
    features.append(int(atom.GetIsAromatic()))
    
    # Number of hydrogens (0-4)
    num_hs = [0] * 5
    total_hs = atom.GetTotalNumHs()
    if total_hs < 5:
        num_hs[total_hs] = 1
    features.extend(num_hs)
    
    # Formal charge (-2 to +2)
    charge = [0] * 5
    fc = atom.GetFormalCharge()
    if -2 <= fc <= 2:
        charge[fc + 2] = 1
    features.extend(charge)
    
    # Chirality
    chiral = [0, 0, 0]  # Unspecified, R, S
    try:
        tag = atom.GetChiralTag()
        if tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
            chiral[1] = 1
        elif tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
            chiral[2] = 1
        else:
            chiral[0] = 1
    except:
        chiral[0] = 1
    features.extend(chiral)
    
    # In ring
    features.append(int(atom.IsInRing()))
    
    return np.array(features, dtype=np.float32)


def bond_features(bond):
    """
    Generate feature vector for a bond.
    
    Returns 6-dimensional feature vector:
    - Bond type (single/double/triple/aromatic)
    - Conjugated (boolean)
    - In ring (boolean)
    """
    features = []
    
    # Bond type (one-hot)
    bond_type = [0, 0, 0, 0]  # Single, double, triple, aromatic
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE:
        bond_type[0] = 1
    elif bt == Chem.rdchem.BondType.DOUBLE:
        bond_type[1] = 1
    elif bt == Chem.rdchem.BondType.TRIPLE:
        bond_type[2] = 1
    elif bt == Chem.rdchem.BondType.AROMATIC:
        bond_type[3] = 1
    features.extend(bond_type)
    
    # Conjugated
    features.append(int(bond.GetIsConjugated()))
    
    # In ring
    features.append(int(bond.IsInRing()))
    
    return np.array(features, dtype=np.float32)


def smiles_to_features(smiles):
    """
    Convert SMILES to atom and bond features.
    
    Args:
        smiles: SMILES string
        
    Returns:
        tuple: (atom_features, edge_index, edge_features)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Get atom features
    atom_feat = np.array([atom_features(atom) for atom in mol.GetAtoms()])
    
    # Get bonds (edge_index and edge_features)
    edge_indices = []
    edge_feat = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        
        # Add both directions (undirected graph)
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_feat.append(bf)
        edge_feat.append(bf)
    
    if len(edge_indices) == 0:
        # Single atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.float)
    
    atom_feat = torch.tensor(atom_feat, dtype=torch.float)
    
    return atom_feat, edge_index, edge_attr
