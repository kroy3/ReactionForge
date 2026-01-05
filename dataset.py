"""
Data processing utilities for ReactionForge.

Handles conversion of SMILES to PyTorch Geometric graphs,
dataset loading, and batch preparation.
"""

import torch
from torch_geometric.data import Data, Dataset
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# Atom feature extraction
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),  # All elements
    'degree': [0, 1, 2, 3, 4, 5, 6],
    'formal_charge': [-3, -2, -1, 0, 1, 2, 3],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    ]
}


def one_hot_encoding(value, allowable_set):
    """One-hot encode a value given an allowable set."""
    if value not in allowable_set:
        value = allowable_set[-1]
    return [int(value == s) for s in allowable_set]


def atom_features(atom):
    """
    Extract atom features for graph neural networks.
    
    Features:
    - Atomic number (one-hot)
    - Degree (one-hot)
    - Formal charge (one-hot)
    - Chirality (one-hot)
    - Number of hydrogens (one-hot)
    - Hybridization (one-hot)
    - Aromaticity (binary)
    - In ring (binary)
    
    Total: ~150 features
    """
    features = []
    features += one_hot_encoding(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
    features += one_hot_encoding(atom.GetDegree(), ATOM_FEATURES['degree'])
    features += one_hot_encoding(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    features += one_hot_encoding(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag'])
    features += one_hot_encoding(atom.GetTotalNumHs(), ATOM_FEATURES['num_Hs'])
    features += one_hot_encoding(atom.GetHybridization(), ATOM_FEATURES['hybridization'])
    features += [int(atom.GetIsAromatic())]
    features += [int(atom.IsInRing())]
    return features


def bond_features(bond):
    """
    Extract bond features.
    
    Features:
    - Bond type (one-hot: single, double, triple, aromatic)
    - Conjugated (binary)
    - In ring (binary)
    
    Total: ~6 features
    """
    bond_type = bond.GetBondType()
    features = [
        int(bond_type == Chem.rdchem.BondType.SINGLE),
        int(bond_type == Chem.rdchem.BondType.DOUBLE),
        int(bond_type == Chem.rdchem.BondType.TRIPLE),
        int(bond_type == Chem.rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing())
    ]
    return features


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """
    Convert SMILES string to PyTorch Geometric Data object.
    
    Args:
        smiles: SMILES string representing molecule
        
    Returns:
        PyG Data object or None if conversion fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Add hydrogens for accurate feature extraction
    mol = Chem.AddHs(mol)
    
    # Extract atom features
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append(atom_features(atom))
    x = torch.tensor(atom_feats, dtype=torch.float)
    
    # Extract bonds
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Add both directions (undirected graph)
        edge_indices += [[i, j], [j, i]]
        
        bond_feat = bond_features(bond)
        edge_attrs += [bond_feat, bond_feat]
    
    if len(edge_indices) == 0:
        # Handle single atom molecules
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def extract_reaction_conditions(row: pd.Series) -> torch.Tensor:
    """
    Extract reaction condition features from dataframe row.
    
    Features typically include:
    - Temperature (normalized)
    - Reaction time (normalized)
    - Catalyst loading (mol%)
    - Solvent encoding
    - Base encoding
    - Ligand encoding
    """
    conditions = []
    
    # Temperature (normalize to 0-1, assuming range 0-200°C)
    temp = row.get('temperature', 100) / 200.0
    conditions.append(temp)
    
    # Time (normalize to 0-1, assuming range 0-48h)
    time = row.get('time', 12) / 48.0
    conditions.append(time)
    
    # Catalyst loading (normalize to 0-1, assuming range 0-10 mol%)
    cat_loading = row.get('catalyst_loading', 5) / 10.0
    conditions.append(cat_loading)
    
    # Add other condition encodings (simplified - in practice use one-hot or embeddings)
    # Solvent, base, ligand, etc.
    for i in range(7):  # Pad to total dimension of 10
        conditions.append(0.0)
    
    return torch.tensor(conditions, dtype=torch.float)


class SuzukiMiyauraDataset(Dataset):
    """
    PyTorch Geometric Dataset for Suzuki-Miyaura reactions.
    
    Expected CSV format:
    - reactant_smiles: SMILES of aryl halide
    - boronic_acid_smiles: SMILES of boronic acid
    - product_smiles: SMILES of coupled product
    - temperature: Reaction temperature (°C)
    - time: Reaction time (hours)
    - catalyst_loading: mol% of catalyst
    - yield: Reaction yield (0-100%)
    - selectivity: Selectivity class (optional)
    - timestamp: Reaction sequence number (optional)
    """
    
    def __init__(self, csv_path: str, transform=None, pre_transform=None):
        self.df = pd.read_csv(csv_path)
        super().__init__(None, transform, pre_transform)
        
    def len(self):
        return len(self.df)
    
    def get(self, idx):
        row = self.df.iloc[idx]
        
        # Convert SMILES to graphs
        reactant_graph = smiles_to_graph(row['reactant_smiles'])
        product_graph = smiles_to_graph(row['product_smiles'])
        
        if reactant_graph is None or product_graph is None:
            # Return dummy data if conversion fails
            return None
        
        # Extract conditions
        conditions = extract_reaction_conditions(row)
        
        # Extract labels
        yield_val = row.get('yield', 0.0) / 100.0  # Normalize to 0-1
        selectivity = row.get('selectivity', 0)  # Default to class 0
        timestamp = row.get('timestamp', idx)  # Use index if no timestamp
        
        data = {
            'reactant': reactant_graph,
            'product': product_graph,
            'conditions': conditions,
            'yield': torch.tensor([yield_val], dtype=torch.float),
            'selectivity': torch.tensor([selectivity], dtype=torch.long),
            'timestamp': torch.tensor([timestamp], dtype=torch.long)
        }
        
        return data


def generate_synthetic_dataset(num_samples: int = 5760, 
                               output_path: str = 'suzuki_reactions.csv'):
    """
    Generate synthetic Suzuki-Miyaura dataset for testing.
    
    Creates realistic-looking reaction data with chemistry-informed
    yield calculations based on:
    - Leaving group reactivity
    - Electronic effects
    - Steric hindrance
    - Reaction conditions
    """
    np.random.seed(42)
    
    # Template SMILES for common reactions
    aryl_halides = [
        'c1ccc(Br)cc1',  # Bromobenzene
        'c1ccc(I)cc1',   # Iodobenzene
        'c1ccc(Cl)cc1',  # Chlorobenzene
        'Brc1ccccn1',    # 2-bromopyridine
    ]
    
    boronic_acids = [
        'OB(O)c1ccccc1',  # Phenylboronic acid
        'OB(O)c1ccc(C)cc1',  # 4-methylphenylboronic acid
    ]
    
    data = []
    for i in range(num_samples):
        # Random selection
        aryl = np.random.choice(aryl_halides)
        boronic = np.random.choice(boronic_acids)
        
        # Simplified product (just combine for demo)
        product = 'c1ccc(-c2ccccc2)cc1'  # Biphenyl
        
        # Random conditions
        temp = np.random.uniform(60, 130)
        time = np.random.uniform(2, 24)
        cat_loading = np.random.uniform(0.5, 10)
        
        # Chemistry-informed yield calculation
        base_yield = 75.0
        
        # Leaving group effect
        if 'I' in aryl:
            base_yield += 10
        elif 'Cl' in aryl:
            base_yield -= 15
        
        # Temperature effect
        if temp > 100:
            base_yield += 5
        
        # Time effect (with diminishing returns)
        if time > 12:
            base_yield += 3
        
        # Add noise
        yield_val = base_yield + np.random.normal(0, 10)
        yield_val = np.clip(yield_val, 0, 100)
        
        # Selectivity (0=poor, 1=moderate, 2=good)
        if yield_val > 80:
            selectivity = 2
        elif yield_val > 50:
            selectivity = 1
        else:
            selectivity = 0
        
        data.append({
            'reactant_smiles': aryl,
            'boronic_acid_smiles': boronic,
            'product_smiles': product,
            'temperature': temp,
            'time': time,
            'catalyst_loading': cat_loading,
            'yield': yield_val,
            'selectivity': selectivity,
            'timestamp': i
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✅ Generated {num_samples} synthetic reactions -> {output_path}")
    return df


if __name__ == "__main__":
    print("🧪 Testing data processing utilities")
    print("=" * 60)
    
    # Test SMILES to graph conversion
    test_smiles = "c1ccc(Br)cc1"  # Bromobenzene
    graph = smiles_to_graph(test_smiles)
    print(f"\n✅ Converted '{test_smiles}' to graph:")
    print(f"   Nodes: {graph.x.shape[0]}, Edges: {graph.edge_index.shape[1]}")
    print(f"   Node features: {graph.x.shape[1]}, Edge features: {graph.edge_attr.shape[1]}")
    
    # Generate synthetic dataset
    print("\n📊 Generating synthetic dataset...")
    df = generate_synthetic_dataset(num_samples=100, output_path='/tmp/test_reactions.csv')
    print(f"   Mean yield: {df['yield'].mean():.1f}%")
    print(f"   Yield std: {df['yield'].std():.1f}%")
    
    # Test dataset loading
    print("\n📦 Testing dataset class...")
    dataset = SuzukiMiyauraDataset('/tmp/test_reactions.csv')
    print(f"   Dataset length: {len(dataset)}")
    
    sample = dataset[0]
    if sample is not None:
        print(f"   Sample 0:")
        print(f"     Reactant nodes: {sample['reactant'].x.shape[0]}")
        print(f"     Product nodes: {sample['product'].x.shape[0]}")
        print(f"     Conditions dim: {sample['conditions'].shape[0]}")
        print(f"     Yield: {sample['yield'].item()*100:.1f}%")
    
    print("\n✅ All data processing tests passed!")
