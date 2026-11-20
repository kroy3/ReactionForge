# ReactionForge User Guide

**Version:** repository snapshot of November 2026
**Audience:** chemists, data scientists, and students who want to train or use ReactionForge to predict Suzuki-Miyaura reaction yields
**Prerequisites:** basic command-line use and basic Python. No deep-learning background is needed for Sections 1–6.

---

## Contents

1. [What ReactionForge Does](#1-what-reactionforge-does)
2. [Project Status: Read This First](#2-project-status-read-this-first)
3. [Key Concepts](#3-key-concepts)
4. [Installation](#4-installation)
5. [Repository Layout](#5-repository-layout)
6. [Preparing Your Data](#6-preparing-your-data)
7. [Quick Start: Train and Predict](#7-quick-start-train-and-predict)
8. [Understanding the Model's Output](#8-understanding-the-models-output)
9. [Configuring the Model](#9-configuring-the-model)
10. [Evaluation and Uncertainty Tools](#10-evaluation-and-uncertainty-tools)
11. [Running the Test Suite](#11-running-the-test-suite)
12. [Troubleshooting](#12-troubleshooting)
13. [Glossary](#13-glossary)
14. [Citation, License, and Support](#14-citation-license-and-support)

---

## 1. What ReactionForge Does

ReactionForge is a deep-learning model that estimates the **yield** (0–100 %) of a **Suzuki-Miyaura cross-coupling reaction**. You give it:

- the structure of the **reactant** (aryl halide) as a SMILES string,
- the structure of the **product** as a SMILES string,
- the **reaction conditions**: temperature, time, and catalyst loading.

It returns:

- a **predicted yield**,
- an **uncertainty estimate** that says how much to trust that prediction, split into *model* uncertainty and *data* uncertainty,
- a **selectivity class** prediction (poor / moderate / good).

Typical uses:

- Ranking candidate reaction conditions before running them in the lab.
- Flagging reactions where the model is unsure and an experiment would be most informative (active learning).
- Research on graph neural networks for chemistry.

The model is built with PyTorch and PyTorch Geometric. It combines graph message passing over the molecular structure (Weisfeiler-Lehman layers), attention between the reactant and product, an optional memory of reaction order (GRU), and an *evidential* output layer that produces uncertainty in a single forward pass.

---

## 2. Project Status: Read This First

The README describes a full pipeline: command-line training, evaluation, batch prediction, hyperparameter search, and pretrained weights. **Not all of this is implemented in the current code.** The table below shows what was confirmed by running the code. It is here so you don't lose time on paths that won't work.

| Component | Status | Notes |
|---|---|---|
| Model (`src/models/reactionforge.py`) | ✅ Works | Bond features must be converted to type indices first. See [§7](#7-quick-start-train-and-predict). |
| Dataset loader (`src/data/dataset.py`) | ✅ Works | Produces 150-dim atom features and 6-dim one-hot bond features |
| Synthetic data generator | ✅ Works | `generate_synthetic_dataset()` |
| Evidential loss function | ✅ Works | `evidential_loss()` |
| Metrics, uncertainty, plotting utilities | ✅ Work | Plain NumPy/matplotlib functions |
| `scripts/train.py` | ❌ Crashes | The trainer calls `model(batch)`, which doesn't match the model's `forward()` signature. The script also hard-codes `node_feat_dim=154`, but the data has 150. |
| `src/training/trainer.py` (`ReactionForgeTrainer`) | ❌ Incompatible with the model | Expects the model to return a `'loss'` key |
| `scripts/evaluate.py`, `scripts/predict.py` | ⚠️ Stubs | They load inputs, then stop at `# Implementation continues...` |
| `scripts/hyperopt.py` | ⚠️ Stub | The objective returns a fixed `0.95` and trains nothing |
| Pretrained checkpoint (`checkpoints/`) | ❌ Not included | `ReactionForge.load_from_checkpoint()` from the README doesn't exist |
| `scripts/run_benchmarks.sh`, `scripts/ablation_study.py`, `docs/*` pages, readthedocs site | ❌ Not in the repository | Referenced in the README only |
| `configs/*.yaml` | ⚠️ Not read by any code | `ablation.yaml` and `transfer_learning.yaml` contain only comments |
| Notebooks (`notebooks/`) | ⚠️ Skeletons | Imports and headings only |
| Unit tests (`tests/`) | ⚠️ 2 of 7 pass | See [§11](#11-running-the-test-suite) |

**About the published numbers.** The performance table in the README (R² = 0.968, RMSE = 5.12 %, and so on) can't be reproduced from this repository alone. The benchmark dataset, training scripts, and weights behind those numbers aren't included. Treat them as claims from the accompanying paper, not as something this code demonstrates.

The rest of this guide documents the path that **does** work, and points out each limitation where it matters.

---

## 3. Key Concepts

You can skip this section if you're already familiar with cheminformatics and machine learning.

**Suzuki-Miyaura coupling.** A palladium-catalysed reaction that joins an aryl halide (for example bromobenzene) with an organoboron compound (for example phenylboronic acid) to form a biaryl (for example biphenyl). Yield depends on the substrates, ligand, base, solvent, temperature, time, and catalyst amount.

**SMILES.** A text notation for molecules. `c1ccc(Br)cc1` is bromobenzene, and `c1ccc(-c2ccccc2)cc1` is biphenyl. Most chemistry software (ChemDraw, RDKit, PubChem) can export SMILES.

**Molecular graph.** ReactionForge turns each SMILES into a graph. Atoms are nodes and bonds are edges. Each atom gets a 150-number feature vector (element, number of neighbours, charge, hydrogens, hybridisation, aromaticity, ring membership). Each bond gets a type: single, double, triple, or aromatic.

**Evidential uncertainty.** The model predicts four numbers per reaction (γ, ν, α, β) instead of one. γ is the predicted yield. The other three describe how confident the model is. From them we get:

- **Aleatoric uncertainty:** noise inherent in the data. Collecting more similar examples won't reduce it.
- **Epistemic uncertainty:** the model's own ignorance, which is high for reactions unlike anything it was trained on. More data *does* reduce it.

**Training, validation, and test sets.** Data is split so the model learns from one part (train), you tune choices on a second part (validation), and you report final performance on a third part it has never seen (test).

---

## 4. Installation

### 4.1 Requirements

| | Minimum |
|---|---|
| Python | 3.10 or newer |
| OS | Linux, macOS, or Windows |
| GPU | Optional. Everything in this guide runs on CPU. |
| Disk | ~3 GB for dependencies |

### 4.2 Install with pip (recommended)

```bash
git clone https://github.com/kroy3/ReactionForge.git
cd ReactionForge

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install torch torch-geometric rdkit pandas numpy scikit-learn tqdm matplotlib seaborn pytest
```

This is the minimal set needed to run the model, data pipeline, and tests. `requirements.txt` lists many more packages (wandb, mordred, plotly, sphinx, nglview, and others) that the current code doesn't import. Install it in full only if you want them:

```bash
pip install -r requirements.txt
```

> **Note on `torch-scatter` / `torch-sparse`.** `requirements.txt` lists these, but recent versions of PyTorch Geometric don't need them, and they often fail to build. If `pip install -r requirements.txt` fails on them, remove those two lines and try again.

### 4.3 Install with conda (GPU, CUDA 11.8)

```bash
conda env create -f environment.yml
conda activate reactionforge
```

This pins Python 3.10, PyTorch 2.0.1, and CUDA 11.8. Use it only if your GPU driver supports CUDA 11.8.

### 4.4 Verify the installation

Run from the repository root:

```bash
python -c "from src.models.reactionforge import ReactionForge; from src.data.dataset import smiles_to_graph; print(smiles_to_graph('c1ccc(Br)cc1'))"
```

Expected output (exact numbers may vary slightly):

```
Data(x=[12, 150], edge_index=[2, 24], edge_attr=[24, 6])
```

> **Always run code from the repository root**, or set `PYTHONPATH` to it. The project isn't an installable package, so `from src...` imports only resolve from there.

---

## 5. Repository Layout

```
ReactionForge/
├── src/                      ← THE CODE YOU SHOULD USE
│   ├── models/
│   │   ├── reactionforge.py  Main model, evidential loss
│   │   ├── wln_layers.py     Alternative WLN layer (not used by the main model)
│   │   └── attention.py      Alternative attention modules (not used by the main model)
│   ├── data/
│   │   ├── dataset.py        CSV → graphs, dataset class, synthetic data generator
│   │   ├── featurization.py  Alternative 144-dim featuriser (not used by dataset.py)
│   │   └── augmentation.py   SMILES enumeration, condition noise
│   ├── training/
│   │   ├── trainer.py        Trainer class (incompatible with model; see §2)
│   │   ├── callbacks.py      EarlyStopping, ModelCheckpoint
│   │   └── metrics.py        R², RMSE, MAE, calibration error
│   └── utils/
│       ├── uncertainty.py    Calibration, active-learning sampling, reliability data
│       └── visualization.py  Prediction and learning-curve plots
├── scripts/                  CLI entry points (see §2 for status)
├── configs/                  YAML files (not read by any code)
├── notebooks/                Tutorial skeletons
├── tests/                    Unit tests
├── figures/                  Paper figures
├── requirements.txt / environment.yml
└── *.py, models/, training/, utils/   ← duplicate copies of src/ files
```

**Duplicate files.** The repository root and the top-level `models/`, `training/`, and `utils/` folders contain byte-for-byte copies of files in `src/`. They're leftovers from upload. **Always import from `src.`**. Editing a root-level copy won't affect anything that imports from `src`.

---

## 6. Preparing Your Data

### 6.1 File format

ReactionForge reads a **CSV file with one reaction per row**. Column names must match exactly.

| Column | Required | Unit / type | Used by the model? | Example |
|---|---|---|---|---|
| `reactant_smiles` | **Yes** | SMILES | Yes | `c1ccc(Br)cc1` |
| `product_smiles` | **Yes** | SMILES | Yes | `c1ccc(-c2ccccc2)cc1` |
| `yield` | **Yes** for training | percent, 0–100 | Yes (target) | `82.5` |
| `temperature` | Recommended | °C | Yes | `90` |
| `time` | Recommended | hours | Yes | `12` |
| `catalyst_loading` | Recommended | mol % | Yes | `5` |
| `selectivity` | Optional | integer class 0 / 1 / 2 | Loaded, but not used in the loss | `2` |
| `timestamp` | Optional | integer ≥ 0, **< 10 000** | Yes (temporal memory) | `137` |
| `boronic_acid_smiles` | Optional | SMILES | **No.** It's read into the table but never featurised. | `OB(O)c1ccccc1` |

Minimal example (`my_reactions.csv`):

```csv
reactant_smiles,product_smiles,temperature,time,catalyst_loading,yield
c1ccc(Br)cc1,c1ccc(-c2ccccc2)cc1,90,12,5,82.5
c1ccc(I)cc1,c1ccc(-c2ccccc2)cc1,80,6,2,91.0
c1ccc(Cl)cc1,c1ccc(-c2ccccc2)cc1,110,24,5,48.3
```

### 6.2 What happens to your data

- **Yield** is divided by 100, so the model works on a 0–1 scale. Predictions come back on the same scale. Multiply by 100 to get a percentage.
- **Temperature** is divided by 200, **time** by 48, and **catalyst loading** by 10. Values outside 0–200 °C, 0–48 h, or 0–10 mol % still work, but they fall outside the range the scaling assumes.
- **Missing condition values** are silently replaced with defaults: 100 °C, 12 h, 5 mol %. Check your CSV for blanks. The model won't warn you.
- **Solvent, base, and ligand are not encoded.** The condition vector has 10 slots. Only the first three are filled, and the remaining seven are always zero. If your yields depend mainly on ligand or base choice, which is common in high-throughput Suzuki datasets, the current model can't see that information.
- **Missing `timestamp`** falls back to the row number. Datasets with more than 10 000 rows and no `timestamp` column will crash (see [§12](#12-troubleshooting)).

### 6.3 Validate your file before training

Invalid SMILES make the loader return `None`, and batching then fails with a confusing error. Check up front:

```python
import pandas as pd
from rdkit import Chem

df = pd.read_csv("my_reactions.csv")
bad = df[df.apply(lambda r: Chem.MolFromSmiles(r.reactant_smiles) is None
                            or Chem.MolFromSmiles(r.product_smiles) is None, axis=1)]
print(f"{len(bad)} invalid rows")
df.drop(bad.index).to_csv("my_reactions_clean.csv", index=False)
```

### 6.4 No data yet? Generate a synthetic set

```python
from src.data.dataset import generate_synthetic_dataset
generate_synthetic_dataset(num_samples=5760, output_path="synthetic.csv")
```

This creates rule-based fake reactions (iodide > bromide > chloride, small bonuses for higher temperature and longer time, plus random noise). It's useful for checking that your setup works. **It isn't real chemistry.** Don't draw scientific conclusions from a model trained on it.

---

## 7. Quick Start: Train and Predict

Because `scripts/train.py` doesn't run (see [§2](#2-project-status-read-this-first)), use the script below instead. It was run end-to-end on CPU against this repository and completes in about a minute.

Save it as `quickstart.py` in the repository root and run `python quickstart.py`.

```python
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from src.models.reactionforge import ReactionForge, evidential_loss
from src.data.dataset import (SuzukiMiyauraDataset, generate_synthetic_dataset,
                              smiles_to_graph, extract_reaction_conditions)


def to_bond_type_index(graph):
    """Convert one-hot bond features (N, 6) into bond-type indices (N,)
    as expected by the model's WLN layers: 0=single 1=double 2=triple 3=aromatic."""
    graph = graph.clone()
    graph.edge_attr = graph.edge_attr[:, :4].argmax(dim=1)
    return graph


# 1. Data  (replace with your own CSV, e.g. "my_reactions_clean.csv")
generate_synthetic_dataset(num_samples=500, output_path="synthetic.csv")
dataset = SuzukiMiyauraDataset("synthetic.csv")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. Model (150 = atom feature size produced by src/data/dataset.py)
model = ReactionForge(node_feat_dim=150, edge_feat_dim=6, condition_dim=10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# 3. Train
for epoch in range(5):
    model.train()
    total = 0.0
    for batch in loader:
        out = model(to_bond_type_index(batch["reactant"]),
                    to_bond_type_index(batch["product"]),
                    batch["conditions"],
                    batch["timestamp"].view(-1))
        gamma, nu, alpha, beta = out["yield_params"]
        loss = evidential_loss(gamma, nu, alpha, beta, batch["yield"].view(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
    print(f"epoch {epoch + 1}: loss {total / len(loader):.4f}")

torch.save({"model_state_dict": model.state_dict()}, "my_model.pt")

# 4. Predict a single reaction
model.eval()
reactant = Batch.from_data_list([smiles_to_graph("c1ccc(Br)cc1")])          # bromobenzene
product = Batch.from_data_list([smiles_to_graph("c1ccc(-c2ccccc2)cc1")])     # biphenyl
conditions = extract_reaction_conditions(
    pd.Series({"temperature": 90, "time": 12, "catalyst_loading": 5})).unsqueeze(0)

with torch.no_grad():
    out = model(to_bond_type_index(reactant), to_bond_type_index(product), conditions)

print(f"Predicted yield: {out['yield_mean'].item() * 100:.1f}%")
print(f"Total uncertainty:     {out['uncertainty'].item():.4f}")
print(f"  epistemic (model):   {out['epistemic_uncertainty'].item():.4f}")
print(f"  aleatoric (data):    {out['aleatoric_uncertainty'].item():.4f}")
```

Sample output (numbers will differ on every run):

```
✅ Generated 500 synthetic reactions -> synthetic.csv
epoch 1: loss 0.5153
epoch 2: loss -0.0408
...
Predicted yield: 41.5%
Total uncertainty:     0.4728
  epistemic (model):   0.1861
  aleatoric (data):    0.2867
```

**Why the loss can be negative.** The evidential loss is a negative log-likelihood, and that can legitimately go below zero. Watch whether it trends downward, not its sign.

**Why `to_bond_type_index` is needed.** The data loader stores each bond as six 0/1 flags. The model's message-passing layer expects one integer per bond (its bond type). Without this conversion you get `IndexError: too many indices for tensor of dimension 1`.

### 7.1 Before using this for real decisions

The quick-start loop is deliberately minimal. For real work, add:

1. **A train/validation/test split.** Use `torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])`. Report metrics on the test set only.
2. **More epochs** with early stopping on validation R². `src.training.callbacks.EarlyStopping(patience=20, mode='max')` works on its own.
3. **Metrics.** Collect `out["yield_mean"]` and `batch["yield"]` over the validation set and pass them to `src.training.metrics.calculate_metrics`.
4. **A split that matches your question.** A random split overestimates how well the model generalises to *new* substrates or ligands. If you want to predict unseen chemistry, hold out whole substrates or ligands.

### 7.2 Loading a saved model

```python
model = ReactionForge(node_feat_dim=150, edge_feat_dim=6, condition_dim=10)  # same args as training
model.load_state_dict(torch.load("my_model.pt")["model_state_dict"])
model.eval()
```

The constructor arguments must exactly match the ones used for training, or `load_state_dict` will fail with a size-mismatch error.

---

## 8. Understanding the Model's Output

`model(...)` returns a dictionary. Each tensor has one row per reaction in the batch.

| Key | Shape | Meaning |
|---|---|---|
| `yield_mean` | (B, 1) | Predicted yield on a 0–1 scale. Multiply by 100 for %. Not clipped, so values can fall slightly outside 0–1. |
| `epistemic_uncertainty` | (B, 1) | Model uncertainty. High means the reaction is unlike the training data. |
| `aleatoric_uncertainty` | (B, 1) | Data noise the model expects for this kind of reaction |
| `uncertainty` | (B, 1) | Sum of the two above |
| `yield_params` | tuple of 4 × (B, 1) | Raw evidential parameters (γ, ν, α, β). Needed for the loss. |
| `selectivity_logits` | (B, 3) | Scores for selectivity classes 0/1/2. Apply `softmax` for probabilities. **Untrained** unless you add a classification loss yourself. |
| `temporal_state` | (1, B, hidden) or `None` | GRU memory. Returned only when `timestamps` are passed. |

**How to read the uncertainty values.**

- They're **variances on the 0–1 yield scale**, not ± percentages. As a rough standard deviation in % points, use `sqrt(uncertainty) * 100`. The README's `uncertainty * 100` shorthand is not a standard deviation.
- Uncalibrated values are mainly useful for **ranking**: which predictions to trust more or less. Before you read them as absolute error bars, calibrate them on a validation set with `src.utils.uncertainty.calibrate_uncertainty` and check them with `reliability_diagram` (see [§10](#10-evaluation-and-uncertainty-tools)).
- High *epistemic* uncertainty is the better signal for "run this experiment next". High *aleatoric* uncertainty means even more data won't make this reaction predictable.

---

## 9. Configuring the Model

All settings are arguments to the `ReactionForge(...)` constructor. The YAML files in `configs/` are **not** read by any code.

| Argument | Default | What it controls | Guidance |
|---|---|---|---|
| `node_feat_dim` | 32 | Size of each atom's feature vector | **Must be 150** with `src/data/dataset.py`. The default of 32 won't work with real data. |
| `edge_feat_dim` | 4 | Bond feature size | Stored but not used by the forward pass. Set it to 6 for consistency. |
| `condition_dim` | 10 | Length of the condition vector | Keep at 10 unless you change `extract_reaction_conditions` |
| `hidden_dim` | 128 | Width of internal layers | 64 (small data) to 256 (large data). Must be divisible by `num_heads`. |
| `num_wln_layers` | 3 | Message-passing depth (how far along a molecule information spreads) | 2–5 |
| `num_heads` | 8 | Attention heads | Must divide `hidden_dim` |
| `pooling_ratio` | 0.5 | Fraction of atoms kept by SAG pooling | Currently has **no effect**: the pooling layer is created but never called |
| `temporal_dim` | 64 | Size of the timestamp encoding | Must be even |
| `dropout` | 0.2 | Regularisation strength | Increase (0.3–0.5) if validation error rises while training error falls |

Model size at the defaults with `node_feat_dim=150` is about 1.03 million parameters. That's lower than the ~3.2 M the tests expect.

**Turning features off.** The README's `--use_temporal_memory` / `--use_cross_attention` flags don't exist. To disable temporal memory, don't pass `timestamps` to `model(...)`. Cross-attention can't be switched off without editing the code.

---

## 10. Evaluation and Uncertainty Tools

These functions are standalone and work on NumPy arrays. Collect predictions first:

```python
import numpy as np

model.eval()
preds, trues, epi, ale = [], [], [], []
with torch.no_grad():
    for batch in test_loader:
        out = model(to_bond_type_index(batch["reactant"]),
                    to_bond_type_index(batch["product"]),
                    batch["conditions"], batch["timestamp"].view(-1))
        preds.append(out["yield_mean"].view(-1)); trues.append(batch["yield"].view(-1))
        epi.append(out["epistemic_uncertainty"].view(-1)); ale.append(out["aleatoric_uncertainty"].view(-1))
y_pred, y_true = torch.cat(preds).numpy(), torch.cat(trues).numpy()
epistemic, aleatoric = torch.cat(epi).numpy(), torch.cat(ale).numpy()
```

Then:

| Function | Module | Purpose |
|---|---|---|
| `calculate_metrics(y_true, y_pred)` | `src.training.metrics` | Returns `{'r2', 'rmse', 'mae'}` on whatever scale you pass. Multiply RMSE/MAE by 100 for % points. |
| `expected_calibration_error(y_true, y_pred, uncertainty)` | `src.training.metrics` | Single calibration score. Lower is better. |
| `calibrate_uncertainty(y_true, y_pred, epistemic, aleatoric)` | `src.utils.uncertainty` | Rescales uncertainties to match observed errors |
| `reliability_diagram(y_true, y_pred, uncertainty)` | `src.utils.uncertainty` | Binned confidence vs. accuracy data for plotting |
| `uncertainty_based_sampling(epistemic, n)` | `src.utils.uncertainty` | Indices of the `n` most uncertain reactions to run next (active learning) |
| `plot_predictions(y_true, y_pred, uncertainty, save_path)` | `src.utils.visualization` | Predicted-vs-true scatter plot |
| `plot_learning_curves(history, save_path)` | `src.utils.visualization` | Training curves |

`src.data.augmentation` provides `enumerate_smiles(smiles, n)` (alternative SMILES spellings of the same molecule) and `perturb_conditions(dict)` (adds small noise to temperature and time). Neither is wired into the dataset. Apply them yourself when building your CSV if you want augmentation.

---

## 11. Running the Test Suite

```bash
python -m pytest tests -q
```

Current expected result: **5 failed, 2 passed**. The failures come from the tests, not from your installation:

| Test | Why it fails |
|---|---|
| `test_atom_features`, `test_smiles_to_features` | Expect 154 features. `featurization.py` produces 144. |
| `test_model_forward` | Calls `model(batch_dict)`, which doesn't match `forward()` |
| `test_model_parameters` | Expects 3–4 M parameters. The default model has ~1 M. |
| `test_early_stopping` | Off-by-one in the test's expectation of when patience runs out |

If you see `ModuleNotFoundError` instead, your installation or working directory is the problem (see [§12](#12-troubleshooting)).

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | Running from outside the repository root | `cd` to the root, or `export PYTHONPATH=/path/to/ReactionForge` |
| `ModuleNotFoundError: No module named 'torch'` / `'rdkit'` | Dependencies not installed in the active environment | Activate your venv or conda env, then re-run [§4.2](#42-install-with-pip-recommended) |
| `IndexError: too many indices for tensor of dimension 1` | One-hot bond features passed straight to the model | Wrap graphs with `to_bond_type_index` ([§7](#7-quick-start-train-and-predict)) |
| `mat1 and mat2 shapes cannot be multiplied (…x150 and 154x128)` or `…and 32x128` | `node_feat_dim` doesn't match the data | Use `node_feat_dim=150` |
| `TypeError: forward() missing 2 required positional arguments` | Running `scripts/train.py` or `ReactionForgeTrainer` | Use the loop in [§7](#7-quick-start-train-and-predict) |
| `TypeError: … NoneType …` during batching | An invalid SMILES in the CSV | Clean the file ([§6.3](#63-validate-your-file-before-training)) |
| `IndexError: index … out of bounds` in `SinusoidalTemporalEncoding` | A `timestamp` (or row number) ≥ 10 000 | Rescale timestamps below 10 000, or omit them at inference |
| `size mismatch for …` on `load_state_dict` | Constructor arguments differ from training | Recreate the model with identical arguments |
| `FileNotFoundError: checkpoints/reactionforge_best.pt` | No pretrained weights ship with the repo | Train your own ([§7](#7-quick-start-train-and-predict)) |
| `torch-scatter` / `torch-sparse` fail to build | Compiled extensions with no matching wheel | Remove them from `requirements.txt`. They aren't needed. |
| Predictions all near the same value | Too little training, or inputs lack informative features (for example, ligand-driven yields) | Train longer, check the split, and review [§6.2](#62-what-happens-to-your-data) |

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **Aleatoric uncertainty** | Irreducible noise in the outcome for a given input |
| **Epistemic uncertainty** | Uncertainty from lack of training data near this input. Reducible. |
| **Evidential regression** | A method where the network outputs the parameters of a distribution (Normal-Inverse-Gamma) rather than a single number, giving uncertainty without ensembles |
| **GRU** | Gated Recurrent Unit, a small recurrent network used here as memory across reaction order |
| **MAE / RMSE** | Mean absolute error / root-mean-square error. Lower is better. |
| **R²** | Fraction of yield variance explained by the model. 1.0 is perfect; 0 means no better than predicting the mean. |
| **SAG pooling** | Self-attention graph pooling: keeps the most important atoms. Defined but currently unused. |
| **SMILES** | Text encoding of a molecule's structure |
| **WLN** | Weisfeiler-Lehman Network, a graph layer that updates each atom from its bonded neighbours, with separate weights per bond type |

---

## 14. Citation, License, and Support

**License:** MIT. See [`LICENSE`](../LICENSE). You may use, modify, and redistribute the code, including commercially, provided the copyright notice is kept.

**Citation:** if you use ReactionForge in research, cite the accompanying preprint as given in the [README](../README.md#-citation). The DOI there is currently a placeholder.

**Reporting problems:** open an issue at <https://github.com/kroy3/ReactionForge/issues> and include:

1. the exact command or script you ran,
2. the full error message,
3. the output of `python -c "import torch, torch_geometric, rdkit; print(torch.__version__, torch_geometric.__version__, rdkit.__version__)"`,
4. a few rows of your CSV, if the problem is data-related.
