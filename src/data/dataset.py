"""
Data processing utilities for ReactionForge.

Handles conversion of SMILES to PyTorch Geometric graphs,
dataset loading, and batch preparation.
"""

import torch
from torch_geometric.data import Data, Dataset
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
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


def _value_or_default(row: pd.Series, column: str, default: float) -> float:
    """Return row[column], or ``default`` if the column is absent or the cell is blank."""
    value = row.get(column, default)
    return default if pd.isna(value) else float(value)


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
    temp = _value_or_default(row, 'temperature', 100) / 200.0
    conditions.append(temp)
    
    # Time (normalize to 0-1, assuming range 0-48h)
    time = _value_or_default(row, 'time', 12) / 48.0
    conditions.append(time)
    
    # Catalyst loading (normalize to 0-1, assuming range 0-10 mol%)
    cat_loading = _value_or_default(row, 'catalyst_loading', 5) / 10.0
    conditions.append(cat_loading)
    
    # Add other condition encodings (simplified - in practice use one-hot or embeddings)
    # Solvent, base, ligand, etc.
    for i in range(7):  # Pad to total dimension of 10
        conditions.append(0.0)
    
    return torch.tensor(conditions, dtype=torch.float)


# Optional reaction components, each given as a SMILES column ``<role>_smiles``.
# Multi-component species (salts) are dot-separated SMILES, e.g. K3PO4 as
# "[K+].[K+].[K+].O=P([O-])([O-])[O-]". A blank cell means "not used"
# (e.g. a ligand-free reaction).
REAGENT_ROLES = ('boronic_acid', 'ligand', 'base', 'solvent')


def reagent_column(role: str) -> str:
    """CSV column holding the SMILES for a reagent role."""
    return f'{role}_smiles'


def detect_reagent_roles(df: pd.DataFrame) -> tuple:
    """Reagent roles that have a column in ``df``, in canonical order."""
    return tuple(r for r in REAGENT_ROLES if reagent_column(r) in df.columns)


def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or \
        (isinstance(value, str) and value.strip() == '')


def _empty_graph(node_dim: int = 150, edge_dim: int = 6) -> Data:
    """Graph with no atoms, used for an absent reagent so batches stay aligned."""
    return Data(x=torch.zeros((0, node_dim)), edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, edge_dim)))


def invalid_smiles_columns(row: pd.Series, reagent_roles=()) -> list:
    """Names of the SMILES columns in ``row`` that RDKit cannot parse.
    Reactant and product are required; reagent columns may be blank."""
    bad = [c for c in ('reactant_smiles', 'product_smiles')
           if _blank(row.get(c)) or Chem.MolFromSmiles(str(row.get(c))) is None]
    for role in reagent_roles:
        value = row.get(reagent_column(role))
        if not _blank(value) and Chem.MolFromSmiles(str(value)) is None:
            bad.append(reagent_column(role))
    return bad


def reagent_inputs(batch: dict, reagent_roles) -> Optional[dict]:
    """
    Pack a collated batch's reagent graphs for ``ReactionForge.forward``.

    Returns ``{role: (Batch, presence mask of shape (B,))}``, or None when the
    model uses no reagents.
    """
    if not reagent_roles:
        return None
    mask = batch['reagent_mask']
    return {role: (batch[f'reagent_{role}'], mask[:, i])
            for i, role in enumerate(reagent_roles)}


class SuzukiMiyauraDataset(Dataset):
    """
    PyTorch Geometric Dataset for Suzuki-Miyaura reactions.

    Expected CSV columns:

    - ``reactant_smiles``: the electrophile (aryl halide or triflate). Required.
    - ``product_smiles``: the coupled product. Required.
    - ``boronic_acid_smiles``, ``ligand_smiles``, ``base_smiles``,
      ``solvent_smiles``: optional reagents (see ``REAGENT_ROLES``); blank
      cells mean the component is absent.
    - ``temperature`` (°C), ``time`` (h), ``catalyst_loading`` (mol%): optional.
    - ``yield``: 0-100 %, needed for training and evaluation.
    - ``selectivity`` (class 0/1/2) and ``timestamp``: optional.

    Args:
        csv_path: path to the CSV, or a DataFrame
        reagent_roles: which reagent roles to load. ``None`` loads every role
            that has a column. A model trained with some roles must be given
            the same roles at prediction time (``load_model`` records them);
            a role missing from the data is then treated as absent.
    """

    def __init__(self, csv_path, transform=None, pre_transform=None, reagent_roles=None):
        # Accept a path or an already-loaded DataFrame
        df = csv_path.copy() if isinstance(csv_path, pd.DataFrame) else pd.read_csv(csv_path)
        if isinstance(csv_path, pd.DataFrame):
            csv_path = 'the input DataFrame'

        self.reagent_roles = (detect_reagent_roles(df) if reagent_roles is None
                              else tuple(reagent_roles))

        # Rows whose SMILES RDKit cannot parse would make get() return None,
        # which crashes batching. Drop them up front and say how many.
        RDLogger.DisableLog('rdApp.*')
        try:
            valid = df.apply(lambda r: not invalid_smiles_columns(r, self.reagent_roles), axis=1)
        finally:
            RDLogger.EnableLog('rdApp.*')
        if (~valid).any():
            print(f"⚠️  Skipping {(~valid).sum()} of {len(df)} rows with "
                  f"invalid SMILES in {csv_path}")
        self.df = df[valid].reset_index(drop=True)
        self._graph_cache = {}

        super().__init__(None, transform, pre_transform)

    def len(self):
        return len(self.df)

    def _graph(self, smiles):
        # Reagent sets are small (a handful of ligands/bases/solvents), so cache
        if smiles not in self._graph_cache:
            self._graph_cache[smiles] = smiles_to_graph(smiles)
        return self._graph_cache[smiles]

    def get(self, idx):
        row = self.df.iloc[idx]

        # Convert SMILES to graphs
        reactant_graph = self._graph(row['reactant_smiles'])
        product_graph = self._graph(row['product_smiles'])

        if reactant_graph is None or product_graph is None:
            # Return dummy data if conversion fails
            return None

        # Extract conditions
        conditions = extract_reaction_conditions(row)

        # Extract labels
        yield_val = _value_or_default(row, 'yield', 0.0) / 100.0  # Normalize to 0-1
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

        if self.reagent_roles:
            present = []
            for role in self.reagent_roles:
                value = row.get(reagent_column(role))
                if _blank(value):
                    data[f'reagent_{role}'] = _empty_graph()
                    present.append(0.0)
                else:
                    data[f'reagent_{role}'] = self._graph(str(value))
                    present.append(1.0)
            data['reagent_mask'] = torch.tensor(present, dtype=torch.float)  # (R,); batches to (B, R)

        return data


# Suzuki-Miyaura coupling: aryl/heteroaryl (pseudo)halide + aryl boronic acid -> biaryl
_SUZUKI_COUPLING = AllChem.ReactionFromSmarts(
    '[c:1][Cl,Br,I,$(OS(=O)(=O)C(F)(F)F)].[c:2][B]>>[c:1]-[c:2]')


_ELECTROPHILE_SITE = Chem.MolFromSmarts('[c][Cl,Br,I,$(OS(=O)(=O)C(F)(F)F)]')
_BORON_SITE = Chem.MolFromSmarts('[c][B]')


def _reacting_fragment(smiles: str, site) -> Chem.Mol:
    """The fragment of a (possibly salt) SMILES that contains the reacting site,
    e.g. the aryltrifluoroborate in 'F[B-](F)(F)c1ccccc1.[K+]'."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    matches = [f for f in Chem.GetMolFrags(mol, asMols=True) if f.HasSubstructMatch(site)]
    if len(matches) != 1:
        raise ValueError(f"Expected one reacting fragment in {smiles!r}, found {len(matches)}")
    return matches[0]


def suzuki_product(electrophile: str, boron_partner: str) -> str:
    """
    Canonical SMILES of the Suzuki-Miyaura product of an aryl (pseudo)halide
    and an aryl boron reagent, e.g. 2-bromopyridine + phenylboronic acid ->
    2-phenylpyridine.
    """
    mols = (_reacting_fragment(electrophile, _ELECTROPHILE_SITE),
            _reacting_fragment(boron_partner, _BORON_SITE))
    products = set()
    for outcome in _SUZUKI_COUPLING.RunReactants(mols):
        mol = outcome[0]
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        products.add(Chem.MolToSmiles(mol))
    if len(products) != 1:
        raise ValueError(f"Expected one coupling product for {electrophile} + "
                         f"{boron_partner}, got {sorted(products)}")
    return products.pop()


# Components used by generate_synthetic_dataset: (SMILES, label)
_SYNTH_ELECTROPHILES = {   # SMILES: leaving group
    'Ic1ccccc1': 'I', 'Brc1ccccc1': 'Br', 'Clc1ccccc1': 'Cl',
    'Brc1ccccn1': 'Br', 'COc1ccc(Br)cc1': 'Br', 'Cc1ccc(Cl)cc1': 'Cl',
    'O=S(=O)(Oc1ccccc1)C(F)(F)F': 'OTf',
}
_SYNTH_BORON = ['OB(O)c1ccccc1', 'Cc1ccc(B(O)O)cc1', 'COc1ccc(B(O)O)cc1']
_SYNTH_LIGANDS = {         # SMILES ('' = no ligand): class
    'c1ccc(P(c2ccccc2)c2ccccc2)cc1': 'simple',                              # PPh3
    'C1CCC(P(C2CCCCC2)C2CCCCC2)CC1': 'bulky',                               # PCy3
    'CC(C)(C)P(C(C)(C)C)C(C)(C)C': 'bulky',                                 # P(tBu)3
    'COc1cccc(OC)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1': 'buchwald',               # SPhos
    'CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1': 'buchwald',  # XPhos
    '': 'none',
}
_SYNTH_BASES = {           # SMILES ('' = no base): yield shift
    '[K+].[K+].[K+].O=P([O-])([O-])[O-]': 8,   # K3PO4
    '[K+].[K+].O=C([O-])[O-]': 4,              # K2CO3
    '[Cs+].[F-]': 2,                           # CsF
    'CCN(CC)CC': -15,                          # Et3N
    '': None,                                  # no base
}
_SYNTH_SOLVENTS = {'C1COCCO1': 3, 'C1CCOC1': 0, 'CN(C)C=O': -2, 'CC#N': -6}  # dioxane, THF, DMF, MeCN


def generate_synthetic_dataset(num_samples: int = 5760,
                               output_path: str = 'suzuki_reactions.csv',
                               seed: int = 42,
                               noise_std: float = 5.0):
    """
    Generate a synthetic Suzuki-Miyaura dataset for testing the pipeline.

    Products are computed from the reactants (``suzuki_product``), so the
    chemistry of each row is self-consistent. Yields follow simple invented
    rules in the direction of textbook trends, not a fitted or validated
    kinetic model:

    - leaving group: I > OTf ~ Br >> Cl
    - aryl chlorides need bulky, electron-rich phosphines (PCy3, P(tBu)3,
      SPhos, XPhos); with PPh3 or no ligand they barely react
    - no base gives almost no product; Et3N is a poor base here
    - small solvent, temperature and time effects, plus Gaussian noise
      (``noise_std`` yield points)

    Use it to check that the code runs and that the model can pick up
    condition effects. Do not draw chemical conclusions from it.
    """
    rng = np.random.default_rng(seed)
    electrophiles = list(_SYNTH_ELECTROPHILES)
    ligands, bases, solvents = list(_SYNTH_LIGANDS), list(_SYNTH_BASES), list(_SYNTH_SOLVENTS)
    product_cache = {}

    data = []
    for i in range(num_samples):
        electrophile = electrophiles[rng.integers(len(electrophiles))]
        boron = _SYNTH_BORON[rng.integers(len(_SYNTH_BORON))]
        ligand = ligands[rng.integers(len(ligands))]
        base = bases[rng.integers(len(bases))]
        solvent = solvents[rng.integers(len(solvents))]
        temp = rng.uniform(60, 120)
        time = rng.uniform(2, 24)
        cat_loading = rng.uniform(0.5, 5)

        key = (electrophile, boron)
        if key not in product_cache:
            product_cache[key] = suzuki_product(electrophile, boron)

        leaving_group = _SYNTH_ELECTROPHILES[electrophile]
        ligand_class = _SYNTH_LIGANDS[ligand]
        base_effect = _SYNTH_BASES[base]

        y = {'I': 82, 'OTf': 74, 'Br': 72, 'Cl': 20}[leaving_group]
        if leaving_group == 'Cl':
            y += {'buchwald': 55, 'bulky': 45, 'simple': 0, 'none': -10}[ligand_class]
        else:
            y += {'buchwald': 10, 'bulky': 6, 'simple': 0, 'none': -25}[ligand_class]
        y += _SYNTH_SOLVENTS[solvent]
        y += 6 * (temp - 90) / 30 + 4 * (min(time, 12) - 7) / 5 + 2 * np.log(cat_loading)
        if base_effect is None:
            y *= 0.1  # transmetalation needs a base
        else:
            y += base_effect
        yield_val = float(np.clip(y + rng.normal(0, noise_std), 0, 100))

        # Selectivity (0=poor, 1=moderate, 2=good)
        selectivity = 2 if yield_val > 80 else 1 if yield_val > 50 else 0

        data.append({
            'reactant_smiles': electrophile,
            'boronic_acid_smiles': boron,
            'ligand_smiles': ligand,
            'base_smiles': base,
            'solvent_smiles': solvent,
            'product_smiles': product_cache[key],
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
