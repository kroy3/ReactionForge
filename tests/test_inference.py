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
    assert list(out['status']) == ['ok', 'invalid SMILES', 'ok']
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


def test_load_model_rejects_checkpoint_without_config(tmp_path, model):
    path = tmp_path / 'old.pt'
    torch.save({'model_state_dict': model.state_dict()}, path)
    with pytest.raises(ValueError, match='model_config'):
        load_model(str(path))
