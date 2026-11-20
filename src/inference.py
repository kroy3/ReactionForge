"""
Inference helpers shared by scripts/predict.py and scripts/evaluate.py.

    from src.inference import load_model, predict_dataframe

    model, use_timestamps = load_model('checkpoints/best_model.pt')
    results = predict_dataframe(model, pd.read_csv('reactions.csv'),
                                use_timestamps=use_timestamps)
"""

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch_geometric.loader import DataLoader

from src.data.dataset import SuzukiMiyauraDataset
from src.models.reactionforge import ReactionForge

# Upper bound of SinusoidalTemporalEncoding's lookup table
MAX_TIMESTAMP = 10000

PREDICTION_COLUMNS = ['predicted_yield', 'uncertainty_std',
                      'epistemic_var', 'aleatoric_var', 'status']


def load_model(checkpoint_path, device='cpu'):
    """
    Rebuild a model from a checkpoint written by scripts/train.py.

    Returns:
        (model in eval mode on ``device``, whether it was trained with timestamps)
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'model_config' not in ckpt:
        raise ValueError(
            f"{checkpoint_path} has no 'model_config'. It was not written by the "
            "current scripts/train.py; retrain, or build ReactionForge(...) with "
            "the original arguments and call load_state_dict yourself.")
    model = ReactionForge(**ckpt['model_config'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model, ckpt.get('use_temporal_memory', True)


def _is_valid(smiles):
    return isinstance(smiles, str) and Chem.MolFromSmiles(smiles) is not None


def predict_dataframe(model, df, use_timestamps=True, batch_size=64, device='cpu'):
    """
    Predict yields for every row of ``df``.

    ``df`` needs ``reactant_smiles`` and ``product_smiles``; condition columns
    (``temperature``, ``time``, ``catalyst_loading``) and ``timestamp`` are used
    when present, exactly as in training.

    Returns:
        DataFrame with the same index as ``df`` and columns
          predicted_yield  - yield in percent
          uncertainty_std  - total predictive standard deviation, percentage points
          epistemic_var    - model (epistemic) variance, 0-1 yield scale
          aleatoric_var    - data (aleatoric) variance, 0-1 yield scale
          status           - 'ok' or the reason the row could not be predicted
        Rows that could not be predicted have NaN in the numeric columns.
    """
    for col in ('reactant_smiles', 'product_smiles'):
        if col not in df.columns:
            raise ValueError(f"Input is missing required column '{col}'")

    out = pd.DataFrame(index=df.index, columns=PREDICTION_COLUMNS, dtype=float)
    out['status'] = 'ok'

    RDLogger.DisableLog('rdApp.*')  # invalid SMILES are reported in 'status'
    try:
        valid = df.apply(lambda r: _is_valid(r['reactant_smiles'])
                         and _is_valid(r['product_smiles']), axis=1)
    finally:
        RDLogger.EnableLog('rdApp.*')
    out.loc[~valid, 'status'] = 'invalid SMILES'

    good = df[valid].reset_index(drop=True)
    if len(good) == 0:
        return out

    if use_timestamps:
        # Same fallback as training: the row position when no timestamp column
        ts = good['timestamp'] if 'timestamp' in good.columns else pd.Series(good.index)
        if ts.max() >= MAX_TIMESTAMP or ts.min() < 0:
            raise ValueError(
                f"Timestamps must be in [0, {MAX_TIMESTAMP}) for this model "
                f"(got {ts.min()}..{ts.max()}). Without a 'timestamp' column the "
                f"row number is used, so split inputs larger than {MAX_TIMESTAMP} "
                f"rows into smaller files.")

    loader = DataLoader(SuzukiMiyauraDataset(good), batch_size=batch_size)
    mean, epi, ale = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            res = model(batch['reactant'], batch['product'], batch['conditions'],
                        batch['timestamp'] if use_timestamps else None)
            mean.append(res['yield_mean'].view(-1).cpu())
            epi.append(res['epistemic_uncertainty'].view(-1).cpu())
            ale.append(res['aleatoric_uncertainty'].view(-1).cpu())

    mean, epi, ale = (torch.cat(t).numpy() for t in (mean, epi, ale))
    idx = df.index[valid]
    out.loc[idx, 'predicted_yield'] = mean * 100
    out.loc[idx, 'uncertainty_std'] = np.sqrt(epi + ale) * 100
    out.loc[idx, 'epistemic_var'] = epi
    out.loc[idx, 'aleatoric_var'] = ale
    return out
