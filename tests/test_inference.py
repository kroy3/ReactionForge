"""Tests for src/inference.py (used by scripts/predict.py and scripts/evaluate.py)"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import extract_reaction_conditions
from src.inference import load_model, predict_dataframe
from src.models.reactionforge import ReactionForge

CONFIG = dict(node_feat_dim=150, edge_feat_dim=6, condition_dim=10,
              hidden_dim=32, num_heads=4)

BIPHENYL = 'c1ccc(-c2ccccc2)cc1'


@pytest.fixture
def model():
    torch.manual_seed(0)
    return ReactionForge(**CONFIG).eval()


def test_predict_dataframe_keeps_rows_aligned(model):
    df = pd.DataFrame({
        'reactant_smiles': ['c1ccc(Br)cc1', 'notasmiles', 'c1ccc(I)cc1'],
        'product_smiles': [BIPHENYL, BIPHENYL, BIPHENYL],
        'temperature': [90, 90, np.nan],  # blank cell -> default, not NaN
    }, index=[10, 20, 30])

    out = predict_dataframe(model, df)

    assert list(out.index) == [10, 20, 30]
    assert list(out['status']) == ['ok', 'invalid SMILES: reactant_smiles', 'ok']
    assert out.loc[[10, 30], 'predicted_yield'].notna().all()
    assert out.loc[20, ['predicted_yield', 'uncertainty_std']].isna().all()
    assert (out.loc[[10, 30], 'uncertainty_std'] > 0).all()


def test_predict_dataframe_requires_smiles_columns(model):
    with pytest.raises(ValueError, match='product_smiles'):
        predict_dataframe(model, pd.DataFrame({'reactant_smiles': ['CC']}))


def test_predict_dataframe_rejects_out_of_range_timestamps(model):
    df = pd.DataFrame({'reactant_smiles': ['c1ccc(Br)cc1'],
                       'product_smiles': [BIPHENYL], 'timestamp': [10000]})
    with pytest.raises(ValueError, match='Timestamps'):
        predict_dataframe(model, df, use_timestamps=True)
    # Fine when the model doesn't use timestamps
    assert predict_dataframe(model, df, use_timestamps=False)['status'].iloc[0] == 'ok'


def test_blank_conditions_use_defaults():
    blank = extract_reaction_conditions(pd.Series({'temperature': np.nan, 'time': np.nan,
                                                   'catalyst_loading': np.nan}))
    missing = extract_reaction_conditions(pd.Series(dtype=float))
    assert torch.isfinite(blank).all()
    assert torch.equal(blank, missing)


def test_load_model_roundtrip(tmp_path, model):
    path = tmp_path / 'best_model.pt'
    torch.save({'model_state_dict': model.state_dict(), 'model_config': CONFIG,
                'use_temporal_memory': False}, path)

    loaded, use_timestamps = load_model(str(path))

    assert use_timestamps is False
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)


def test_predict_dataframe_applies_uncertainty_scale(model):
    df = pd.DataFrame({'reactant_smiles': ['c1ccc(Br)cc1'], 'product_smiles': [BIPHENYL]})

    raw = predict_dataframe(model, df, uncertainty_scale=1.0)
    scaled = predict_dataframe(model, df, uncertainty_scale=0.5)

    assert scaled['predicted_yield'].iloc[0] == pytest.approx(raw['predicted_yield'].iloc[0])
    assert scaled['uncertainty_std'].iloc[0] == pytest.approx(0.5 * raw['uncertainty_std'].iloc[0])
    assert scaled['epistemic_var'].iloc[0] == pytest.approx(0.25 * raw['epistemic_var'].iloc[0])


def test_load_model_reads_uncertainty_scale(tmp_path, model):
    path = tmp_path / 'calibrated.pt'
    torch.save({'model_state_dict': model.state_dict(), 'model_config': CONFIG,
                'uncertainty_scale': 0.4}, path)
    loaded, _ = load_model(str(path))
    assert loaded.uncertainty_scale == pytest.approx(0.4)
    assert loaded.uncertainty_calibrated

    # Checkpoints from before calibration load as uncalibrated (scale 1)
    old = tmp_path / 'old.pt'
    torch.save({'model_state_dict': model.state_dict(), 'model_config': CONFIG}, old)
    loaded, _ = load_model(str(old))
    assert loaded.uncertainty_scale == 1.0
    assert not loaded.uncertainty_calibrated


def test_load_model_rejects_checkpoint_without_config(tmp_path, model):
    path = tmp_path / 'old.pt'
    torch.save({'model_state_dict': model.state_dict()}, path)
    with pytest.raises(ValueError, match='model_config'):
        load_model(str(path))


def test_predict_dataframe_with_reagents():
    """A model trained with reagents reads them, reports a bad reagent SMILES
    by column, and treats a reagent column missing from the input as absent."""
    torch.manual_seed(0)
    roles = ('boronic_acid', 'ligand', 'base')
    model = ReactionForge(**CONFIG, reagent_roles=roles).eval()
    df = pd.DataFrame({
        'reactant_smiles': ['c1ccc(Cl)cc1', 'c1ccc(Cl)cc1', 'c1ccc(Cl)cc1'],
        'product_smiles': [BIPHENYL] * 3,
        'boronic_acid_smiles': ['OB(O)c1ccccc1'] * 3,
        'ligand_smiles': ['C1CCC(P(C2CCCCC2)C2CCCCC2)CC1', None, 'notasmiles'],
        # no base_smiles column at all
    })

    out = predict_dataframe(model, df, use_timestamps=False)

    assert list(out['status']) == ['ok', 'ok', 'invalid SMILES: ligand_smiles']
    # PCy3 vs no ligand must change the prediction
    assert out.loc[0, 'predicted_yield'] != pytest.approx(out.loc[1, 'predicted_yield'])
