"""Tests for reagent (boron partner / ligand / base / solvent) handling and
the Suzuki product helper"""

import importlib.util
import os

import pandas as pd
import pytest
import torch
from torch_geometric.loader import DataLoader

from src.data.dataset import (SuzukiMiyauraDataset, generate_synthetic_dataset,
                              reagent_inputs, suzuki_product)
from src.models.reactionforge import ReactionForge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize('electrophile, boron, product', [
    ('Brc1ccccn1', 'OB(O)c1ccccc1', 'c1ccc(-c2ccccn2)cc1'),              # 2-phenylpyridine
    ('Clc1ccccc1', 'Cc1ccc(B(O)O)cc1', 'Cc1ccc(-c2ccccc2)cc1'),
    ('O=S(=O)(Oc1ccccc1)C(F)(F)F', 'OB(O)c1ccccc1', 'c1ccc(-c2ccccc2)cc1'),  # triflate
    ('Brc1ccccc1', 'F[B-](F)(F)c1ccccn1.[K+]', 'c1ccc(-c2ccccn2)cc1'),       # BF3K salt
    ('Brc1ccccc1', 'CC1(C)OB(c2ccccc2)OC1(C)C', 'c1ccc(-c2ccccc2)cc1'),     # pinacol ester
])
def test_suzuki_product(electrophile, boron, product):
    assert suzuki_product(electrophile, boron) == product


def test_suzuki_product_rejects_non_coupling_partners():
    with pytest.raises(ValueError):
        suzuki_product('c1ccccc1', 'OB(O)c1ccccc1')  # no leaving group


def test_synthetic_products_match_reactants(tmp_path):
    df = generate_synthetic_dataset(200, str(tmp_path / 's.csv'))
    for _, r in df.iterrows():
        assert r['product_smiles'] == suzuki_product(r['reactant_smiles'], r['boronic_acid_smiles'])


def test_synthetic_ligand_effect_on_chlorides(tmp_path):
    """Built-in trend: bulky biaryl phosphines rescue aryl chlorides, PPh3 doesn't."""
    df = generate_synthetic_dataset(3000, str(tmp_path / 's.csv'))
    chlorides = df[df['reactant_smiles'].str.contains('Cl')]
    xphos = chlorides[chlorides['ligand_smiles'].str.contains('C(C)C)c(-c2', regex=False, na=False)]
    pph3 = chlorides[chlorides['ligand_smiles'] == 'c1ccc(P(c2ccccc2)c2ccccc2)cc1']
    assert xphos['yield'].mean() > pph3['yield'].mean() + 25


def test_dataset_reagent_masks_and_validation(tmp_path):
    csv = tmp_path / 'r.csv'
    pd.DataFrame({
        'reactant_smiles': ['c1ccc(Br)cc1'] * 3,
        'product_smiles': ['c1ccc(-c2ccccc2)cc1'] * 3,
        'boronic_acid_smiles': ['OB(O)c1ccccc1'] * 3,
        'ligand_smiles': ['c1ccc(P(c2ccccc2)c2ccccc2)cc1', '', 'notasmiles'],
        'yield': [80, 40, 10],
    }).to_csv(csv, index=False)

    ds = SuzukiMiyauraDataset(str(csv))

    assert ds.reagent_roles == ('boronic_acid', 'ligand')
    assert len(ds) == 2  # invalid ligand row dropped
    batch = next(iter(DataLoader(ds, batch_size=2)))
    assert batch['reagent_mask'].tolist() == [[1.0, 1.0], [1.0, 0.0]]
    assert batch['reagent_ligand'].num_graphs == 2


def test_reagent_free_model_keeps_original_parameters():
    """Checkpoints trained before reagent support must still load."""
    old_style = ReactionForge(node_feat_dim=150, edge_feat_dim=6, condition_dim=10)
    new_style = ReactionForge(node_feat_dim=150, edge_feat_dim=6, condition_dim=10,
                              reagent_roles=('ligand',))
    assert not any(k.startswith('reagent') for k in old_style.state_dict())
    assert any(k.startswith('reagent_proj.ligand') for k in new_style.state_dict())
    ReactionForge(node_feat_dim=150, edge_feat_dim=6, condition_dim=10).load_state_dict(
        old_style.state_dict())


def test_model_uses_reagents(tmp_path):
    """Changing only the ligand changes the prediction; a fully absent role in a
    batch (trailing empty graphs) still yields one output per reaction."""
    torch.manual_seed(0)
    df = pd.DataFrame({
        'reactant_smiles': ['Clc1ccccc1'] * 3,
        'product_smiles': ['c1ccc(-c2ccccc2)cc1'] * 3,
        'ligand_smiles': ['c1ccc(P(c2ccccc2)c2ccccc2)cc1', 'C1CCC(P(C2CCCCC2)C2CCCCC2)CC1', None],
        'base_smiles': [None, None, None],
    })
    ds = SuzukiMiyauraDataset(df)
    model = ReactionForge(node_feat_dim=150, edge_feat_dim=6, condition_dim=10,
                          reagent_roles=ds.reagent_roles).eval()
    batch = next(iter(DataLoader(ds, batch_size=3)))
    with torch.no_grad():
        out = model(batch['reactant'], batch['product'], batch['conditions'],
                    reagents=reagent_inputs(batch, ds.reagent_roles))['yield_mean'].view(-1)
    assert out.shape == (3,)
    assert len(set(round(v, 6) for v in out.tolist())) == 3


def _load_prepare_script():
    spec = importlib.util.spec_from_file_location(
        'prepare_suzuki_dataset', os.path.join(ROOT, 'scripts', 'prepare_suzuki_dataset.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_suzuki_dataset_convert():
    """Both coupling orientations of the Perera data map to the same product,
    with the electrophile/boron roles assigned correctly (no download)."""
    prep = _load_prepare_script()
    xlsx = pd.DataFrame({
        'Reactant_1_Short_Hand': ['1a, 6-Cl-Q', '1g, 6-BF3K-Q'],
        'Reactant_2_Name': ['2b, Boronic Ester', '2d, Bromide'],
        'Ligand_Short_Hand': ['XPhos', float('nan')],
        'Reagent_1_Short_Hand': ['K3PO4', 'None'],
        'Solvent_1_Short_Hand': ['THF_V2', 'MeOH/H2O_V2 9:1'],
        'Product_Yield_PCT_Area_UV': [55.0, 12.5],
    })

    out = prep.convert(xlsx)

    assert list(out['reactant_smiles']) == ['Clc1ccc2ncccc2c1', 'Cc1ccc2c(cnn2C2CCCCO2)c1Br']
    assert out.loc[1, 'boronic_acid_smiles'] == 'F[B-](F)(F)c1ccc2ncccc2c1.[K+]'
    assert set(out['product_smiles']) == {prep.EXPECTED_PRODUCT}
    assert list(out['ligand_smiles']) == [prep.LIGANDS['XPhos'], '']
    assert list(out['base_smiles']) == [prep.BASES['K3PO4'], '']
    assert list(out['solvent_smiles']) == ['C1CCOC1', 'CO']
    assert list(out['yield']) == [55.0, 12.5]
