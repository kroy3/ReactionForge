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

- the structure of the **electrophile** (aryl halide or triflate) and the **product**, as SMILES strings,
- optionally, the **boron coupling partner**, **ligand**, **base** and **solvent**, also as SMILES,
- optionally, **temperature, time and catalyst loading**.

It returns:

- a **predicted yield**,
- an **uncertainty estimate** that says how much to trust that prediction, split into *model* uncertainty and *data* uncertainty,
- a **selectivity class** prediction (poor / moderate / good).

Typical uses:

- Ranking candidate reaction conditions before running them in the lab.
- Flagging reactions where the model is unsure and an experiment would be most informative (active learning).
- Research on graph neural networks for chemistry.

The model is built with PyTorch and PyTorch Geometric. Every molecule (electrophile, product, boron partner, ligand, base, solvent) is turned into a graph of atoms and bonds and encoded with the same message-passing layers (Weisfeiler-Lehman network). It then combines these with attention between the electrophile and product, an optional memory of reaction order (GRU), and an *evidential* output layer that produces uncertainty in a single forward pass.

Because ligands, bases and solvents are read as *structures* rather than as names from a fixed list, the model can in principle make predictions for a ligand it hasn't seen. How well that works depends on how similar the new ligand is to those in the training data. The epistemic uncertainty should be higher in that case, and you should check it.

---

## 2. Project Status: Read This First

The README describes a full pipeline: command-line training, evaluation, batch prediction, hyperparameter search, and pretrained weights. **Not all of this is implemented in the current code.** The table below shows what was confirmed by running the code. It is here so you don't lose time on paths that won't work.

| Component | Status | Notes |
|---|---|---|
| Model (`src/models/reactionforge.py`) | ✅ Works | |
| Dataset loader (`src/data/dataset.py`) | ✅ Works | Produces 150-dim atom features and 6-dim one-hot bond features |
| Synthetic data generator | ✅ Works | `generate_synthetic_dataset()`; products computed from reactants, invented yield rules ([§6.4](#64-no-data-yet-generate-a-synthetic-set)) |
| Ligand / base / solvent / boron partner | ✅ Encoded | As molecular graphs, from optional CSV columns ([§6.1](#61-file-format)) |
| `scripts/prepare_suzuki_dataset.py` | ✅ Works | Downloads and converts the 5,760-reaction Perera et al. dataset ([§7.7](#77-benchmark-on-real-data-the-perera-et-al-suzuki-dataset)) |
| Evidential loss function | ✅ Works | `evidential_loss()` |
| Metrics, uncertainty, plotting utilities | ✅ Work | Plain NumPy/matplotlib functions |
| `scripts/train.py` | ✅ Works | Train / validate / test with early stopping and checkpointing. See [§7.1](#71-train-from-the-command-line). |
| `src/training/trainer.py` (`ReactionForgeTrainer`) | ✅ Works | Used by `scripts/train.py`; also usable from Python ([§7.5](#75-train-from-python)) |
| `scripts/predict.py` | ✅ Works | Batch prediction from a CSV. See [§7.2](#72-predict-many-reactions-from-the-command-line). |
| `scripts/evaluate.py` | ✅ Works | Scores a model on labelled data. See [§7.3](#73-evaluate-on-held-out-data). |
| `scripts/hyperopt.py` | ⚠️ Stub | The objective returns a fixed `0.95` and trains nothing |
| Pretrained checkpoint (`checkpoints/`) | ❌ Not included | Train your own with `scripts/train.py`. Load it with `src.inference.load_model` ([§7.4](#74-predict-from-python)). |
| `scripts/run_benchmarks.sh`, `scripts/ablation_study.py` | ❌ Not in the repository | Referenced in the README only |
| Documentation site | ✅ Builds | This guide plus an API reference, built by Read the Docs from `docs/` ([§11](#11-running-the-test-suite)) |
| `configs/*.yaml` | ⚠️ Not read by any code | `ablation.yaml` and `transfer_learning.yaml` contain only comments |
| Notebooks (`notebooks/`) | ⚠️ Skeletons | Imports and headings only |
| Unit tests (`tests/`) | ✅ All pass | Also run automatically on GitHub for every pull request. See [§11](#11-running-the-test-suite). |

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
│   │   ├── trainer.py        Training loop used by scripts/train.py
│   │   ├── callbacks.py      EarlyStopping, ModelCheckpoint
│   │   └── metrics.py        R², RMSE, MAE, calibration error
│   ├── utils/
│   │   ├── uncertainty.py    Calibration, active-learning sampling, reliability data
│   │   └── visualization.py  Prediction and learning-curve plots
│   └── inference.py          load_model, predict_dataframe (used by predict/evaluate)
├── scripts/                  CLI entry points (see §2 for status)
├── configs/                  YAML files (not read by any code)
├── notebooks/                Tutorial skeletons
├── tests/                    Unit tests
├── docs/                     This guide, API reference page, Sphinx config (Read the Docs)
├── figures/                  Paper figures
├── requirements.txt / environment.yml
└── *.py, models/, training/, utils/   ← duplicate copies of src/ files
```

**Duplicate files.** The repository root and the top-level `models/`, `training/`, and `utils/` folders contain older copies of files in `src/`. They're leftovers from upload, aren't imported by anything, and don't include later fixes. **Always import from `src.`**. Editing a root-level copy won't affect anything.

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
| `boronic_acid_smiles` | Recommended | SMILES | Yes | `OB(O)c1ccccc1` |
| `ligand_smiles` | Recommended | SMILES; blank = no ligand | Yes | `C1CCC(P(C2CCCCC2)C2CCCCC2)CC1` (PCy₃) |
| `base_smiles` | Recommended | SMILES; blank = no base | Yes | `[K+].[K+].[K+].O=P([O-])([O-])[O-]` (K₃PO₄) |
| `solvent_smiles` | Recommended | SMILES | Yes | `C1CCOC1` (THF) |
| `selectivity` | Optional | integer class 0 / 1 / 2 | Loaded, but not used in the loss | `2` |
| `timestamp` | Optional | integer ≥ 0, **< 10 000** | Yes (temporal memory) | `137` |

**The reagent columns** (`boronic_acid_smiles`, `ligand_smiles`, `base_smiles`, `solvent_smiles`):

- **Salts** are written as dot-separated ions, as in the K₃PO₄ example. Boronate salts work too, e.g. `F[B-](F)(F)c1ccccc1.[K+]` for a potassium aryltrifluoroborate.
- **A blank cell means the component is absent** (a ligand-free or base-free reaction). The model learns a separate "absent" representation, so leave the cell blank rather than inventing a placeholder.
- **A model uses the reagent columns it was trained with.** `scripts/train.py` uses every reagent column present in the training CSV (`--no_reagents` turns this off). At prediction time the same columns are read; if one is missing from the prediction CSV, that component is treated as absent for every row, which is usually not what you want.
- **Catalyst:** there is no catalyst column. Datasets that vary the palladium source can add it by including it in the ligand SMILES (e.g. `ligand.[Pd]`), but this hasn't been tested.

Example (`my_reactions.csv`; the yields are illustrative):

```text
reactant_smiles,boronic_acid_smiles,ligand_smiles,base_smiles,solvent_smiles,product_smiles,temperature,yield
c1ccc(Br)cc1,OB(O)c1ccccc1,c1ccc(P(c2ccccc2)c2ccccc2)cc1,[K+].[K+].O=C([O-])[O-],C1COCCO1,c1ccc(-c2ccccc2)cc1,90,82.5
c1ccc(Cl)cc1,OB(O)c1ccccc1,c1ccc(P(c2ccccc2)c2ccccc2)cc1,[K+].[K+].O=C([O-])[O-],C1COCCO1,c1ccc(-c2ccccc2)cc1,90,6.1
c1ccc(Cl)cc1,OB(O)c1ccccc1,COc1cccc(OC)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1,[K+].[K+].[K+].O=P([O-])([O-])[O-],C1COCCO1,c1ccc(-c2ccccc2)cc1,100,88.0
```

### 6.2 What happens to your data

- **Yield** is divided by 100, so the model works on a 0–1 scale. Predictions come back on the same scale. Multiply by 100 to get a percentage.
- **Temperature** is divided by 200, **time** by 48, and **catalyst loading** by 10. Values outside 0–200 °C, 0–48 h, or 0–10 mol % still work, but they fall outside the range the scaling assumes.
- **Missing condition values** (a missing column or a blank cell) are silently replaced with defaults: 100 °C, 12 h, 5 mol %. Check your CSV for blanks. The model won't warn you.
- **Reagents are encoded as molecular graphs**, using the same encoder as the electrophile, so structurally similar ligands get similar representations. The separate 10-slot condition vector still holds only temperature, time and catalyst loading (the other 7 slots are zero).
- **Missing `timestamp`** falls back to the row number. Datasets with more than 10 000 rows and no `timestamp` column will crash (see [§12](#12-troubleshooting)).

### 6.3 Validate your file before training

`SuzukiMiyauraDataset` drops rows whose SMILES RDKit can't parse and prints how many it dropped. If more than a handful are dropped, find and fix them rather than lose the data:

```python
import pandas as pd
from src.data.dataset import detect_reagent_roles, invalid_smiles_columns

df = pd.read_csv("my_reactions.csv")
bad = df.apply(lambda r: invalid_smiles_columns(r, detect_reagent_roles(df)), axis=1)
print(df[bad.apply(len) > 0].assign(bad_columns=bad))
```

It's also worth checking that each product really is the coupling product of its reactants. A mismatch usually means a copy-paste error in the spreadsheet:

```python
from src.data.dataset import suzuki_product

def matches(r):
    try:
        return suzuki_product(r.reactant_smiles, r.boronic_acid_smiles) == r.product_smiles
    except ValueError:
        return False  # not a recognisable aryl (pseudo)halide + aryl boron pair

print(df[~df.apply(matches, axis=1)])
```

`suzuki_product` returns RDKit's canonical SMILES, so compare against canonical SMILES. Convert yours with `Chem.MolToSmiles(Chem.MolFromSmiles(s))` first if needed.

### 6.4 No data yet? Generate a synthetic set

```python
from src.data.dataset import generate_synthetic_dataset
generate_synthetic_dataset(num_samples=5760, output_path="synthetic.csv")
```

This creates rule-based reactions with every column filled in. Each product is computed from its reactants with an RDKit reaction template (`suzuki_product`), so 2-bromopyridine + phenylboronic acid gives 2-phenylpyridine. Yields follow simple invented rules in the direction of textbook trends:

- leaving group: I > OTf ≈ Br ≫ Cl;
- aryl chlorides react well only with bulky, electron-rich phosphines (PCy₃, P(tBu)₃, SPhos, XPhos);
- without a base there is almost no product, and Et₃N is a poor base here;
- small solvent, temperature and time effects, plus random noise (5 yield points by default).

It's useful for checking that your setup works and that the model picks up condition effects. **The yield rules are invented, not fitted to experiments.** Don't draw chemical conclusions from a model trained on it. For real data, see [§7.7](#77-benchmark-on-real-data-the-perera-et-al-suzuki-dataset).

---

## 7. Quick Start: Train and Predict

### 7.1 Train from the command line

Run from the repository root (or give the full path to the script):

```bash
python scripts/train.py \
    --data_path my_reactions.csv \
    --output_dir checkpoints/experiment_001 \
    --epochs 200
```

No data yet? Generate a synthetic set first ([§6.4](#64-no-data-yet-generate-a-synthetic-set)).

What the script does:

1. Loads the CSV, skipping any rows with unparseable SMILES (it prints how many it skipped).
2. Splits the data 70 / 15 / 15 into train / validation / test. The split is reproducible for a given `--seed`.
3. Trains, printing validation R², RMSE and MAE (in yield percentage points) after each epoch.
4. Saves the best model by validation R² to `<output_dir>/best_model.pt`, and stops early once validation R² hasn't improved for `--patience` epochs.
5. Reloads the best model and **calibrates its uncertainty** on the validation set: it fits one scale factor that makes the predicted error bars match the errors the model actually makes, and stores it in the checkpoint ([§8.1](#81-uncertainty-calibration)).
6. Reports scores on the held-out test set, including how well the uncertainty is calibrated there before and after scaling.

Excerpt from a real run on 2 000 synthetic reactions (CPU, about 2 minutes):

```
Split: 1400 train, 300 val, 300 test
Model parameters: 1,028,873
...
Epoch 26/40
Val R²: 0.4830, RMSE: 9.60%, MAE: 7.89%
Early stopping at epoch 26
============================================================
FINAL TEST EVALUATION
============================================================
Test R²: 0.3370
Test RMSE: 10.74%
Test MAE: 8.65%

Uncertainty calibration (scale 0.375, fitted on validation set)
Test reactions within ±1 / ±2 std (ideal ≈ 68% / 95%):
  before: 99% / 100%
  after:  61% / 91%
```

The synthetic generator adds random noise with a standard deviation of 10 yield points, so an RMSE near 10 % is about as good as any model can do on that data. Don't compare these numbers with the paper's.

**Command-line options**

| Option | Default | Meaning |
|---|---|---|
| `--data_path` | *(required)* | Reaction CSV ([§6](#6-preparing-your-data)) |
| `--output_dir` | `checkpoints/` | Where `best_model.pt` is written (created if missing) |
| `--epochs` | 200 | Maximum number of passes over the training data |
| `--patience` | 20 | Epochs without validation improvement before stopping early |
| `--batch_size` | 64 | Reactions per training step |
| `--learning_rate` | 1e-3 | Adam learning rate (halved automatically when validation R² stalls) |
| `--weight_decay` | 1e-5 | L2 regularisation |
| `--hidden_dim` | 128 | Model width ([§9](#9-configuring-the-model)) |
| `--num_wln_layers` | 3 | Message-passing depth |
| `--num_heads` | 8 | Attention heads (must divide `--hidden_dim`) |
| `--dropout` | 0.2 | Dropout rate |
| `--evidential_lambda` | 0.01 | Weight of the uncertainty regulariser in the loss |
| `--no_temporal_memory` | off | Ignore the `timestamp` column (disables the GRU memory) |
| `--no_reagents` | off | Ignore the boron partner / ligand / base / solvent columns |
| `--val_fraction` | 0.15 | Share of the data used for validation (early stopping and uncertainty calibration) |
| `--test_fraction` | 0.15 | Share held out for the final test report. Set to 0 when you have a separate test file to score with `evaluate.py`. |
| `--num_workers` | 0 | Background processes for data loading |
| `--seed` | 42 | Random seed for weights and the data split |

**Why the training loss can be negative.** The evidential loss is a negative log-likelihood, which can legitimately go below zero. Judge training by validation R² and RMSE, not by the sign of the loss.

### 7.2 Predict many reactions from the command line

Put the reactions you want scored in a CSV with the same columns as your training data. The `yield` column isn't needed.

```bash
python scripts/predict.py \
    --checkpoint checkpoints/experiment_001/best_model.pt \
    --input_csv candidates.csv \
    --output_csv predictions.csv
```

`predictions.csv` is your input file with these columns added. Rows stay in the original order:

| Column | Meaning |
|---|---|
| `predicted_yield` | Predicted yield, in percent |
| `uncertainty_std` | Calibrated predictive standard deviation, in percentage points. A reaction predicted at 70 % with `uncertainty_std` 8 is a "70 ± 8 %" prediction: about 2 in 3 such reactions should land within that range ([§8.1](#81-uncertainty-calibration)). |
| `epistemic_var` | Model uncertainty (variance, 0–1 scale). Highest for reactions unlike the training data. Use it to choose experiments. |
| `aleatoric_var` | Expected data noise (variance, 0–1 scale) |
| `status` | `ok`, or why the row couldn't be predicted, naming the column (for example `invalid SMILES: ligand_smiles`). Such rows have empty prediction columns. |

To rank candidate conditions, sort by `predicted_yield`. To choose the most informative next experiments, sort by `epistemic_var`.

### 7.3 Evaluate on held-out data

To measure how good a trained model is, score it on reactions **it wasn't trained on**. These must have a `yield` column.

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/experiment_001/best_model.pt \
    --data_path held_out.csv \
    --output_csv evaluation.csv \
    --plot parity.png
```

`--output_csv` (per-reaction predictions plus `abs_error`) and `--plot` (a predicted-vs-true scatter plot coloured by uncertainty) are optional.

Example output, from a model trained on 2 000 synthetic reactions and scored on 1 000 different ones:

```
Reactions scored: 1000
R²:   0.4279
RMSE: 9.76%
MAE:  7.82%
ECE:  0.0213
Uncertainty calibration: scale 0.375 (fitted on the validation set during training)
Uncertainty coverage (ideal for a calibrated Gaussian: 68% / 95%):
  true yield within ±1 std: 67%
  true yield within ±2 std: 96%
```

**How to read the coverage lines.** If the uncertainty is well calibrated, about 68 % of true yields fall within ±1 standard deviation of the prediction, and 95 % within ±2. The run above is close to both. Much more than 68 % means the error bars are too wide; well under 68 % means the model is overconfident. Calibration is fitted on your training data's validation split, so it can drift on chemistry unlike that data. Check coverage on held-out data before quoting `uncertainty_std` as an error bar.

**Don't evaluate on your training CSV.** `scripts/train.py` already reports scores on its own held-out test split. Running `evaluate.py` on the full training file mixes in reactions the model has seen, which inflates the scores.

### 7.4 Predict from Python

The checkpoint stores the settings the model was built with (`model_config`), and `load_model` rebuilds it from them. For a table of reactions, `predict_dataframe` returns the same columns as `scripts/predict.py`:

```python
import pandas as pd
from src.inference import load_model, predict_dataframe

model, use_timestamps = load_model("checkpoints/experiment_001/best_model.pt")
preds = predict_dataframe(model, pd.read_csv("candidates.csv"), use_timestamps=use_timestamps)
```

For one reaction at a time, with the model's raw outputs:

```python
import torch
import pandas as pd
from torch_geometric.data import Batch

from src.inference import load_model
from src.data.dataset import smiles_to_graph, extract_reaction_conditions

model, _ = load_model("checkpoints/experiment_001/best_model.pt")

reactant = Batch.from_data_list([smiles_to_graph("c1ccc(Br)cc1")])          # bromobenzene
product = Batch.from_data_list([smiles_to_graph("c1ccc(-c2ccccc2)cc1")])     # biphenyl
conditions = extract_reaction_conditions(
    pd.Series({"temperature": 90, "time": 12, "catalyst_loading": 5})).unsqueeze(0)

with torch.no_grad():
    out = model(reactant, product, conditions)

print(f"Predicted yield: {out['yield_mean'].item() * 100:.1f}%")
print(f"Total uncertainty:     {out['uncertainty'].item():.4f}")
print(f"  epistemic (model):   {out['epistemic_uncertainty'].item():.4f}")
print(f"  aleatoric (data):    {out['aleatoric_uncertainty'].item():.4f}")
```

To predict several reactions at once, put several graphs in each `Batch.from_data_list([...])` and stack their condition vectors with `torch.stack`. [§8](#8-understanding-the-models-output) explains the outputs.

### 7.5 Train from Python

To customise the loop (for example a different split), use the trainer directly. This is what `scripts/train.py` does internally:

```python
import torch
from torch_geometric.loader import DataLoader

from src.models.reactionforge import ReactionForge
from src.data.dataset import SuzukiMiyauraDataset
from src.training.trainer import ReactionForgeTrainer

dataset = SuzukiMiyauraDataset("my_reactions.csv")
train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

model = ReactionForge(node_feat_dim=dataset[0]["reactant"].x.shape[1],
                      edge_feat_dim=6, condition_dim=10)
trainer = ReactionForgeTrainer(model, torch.optim.Adam(model.parameters(), lr=1e-3),
                               device="cpu")   # or "cuda"

for epoch in range(50):
    train_loss = trainer.train_epoch(train_loader)
    metrics = trainer.evaluate(val_loader)  # {'loss', 'r2', 'rmse', 'mae'}; rmse/mae in % points
    print(epoch, round(train_loss, 4), metrics)
```

### 7.6 Before using this for real decisions

**Choose a split that matches your question.** The default random split overestimates how well the model generalises to *new* substrates or ligands, because near-identical reactions end up in both training and test sets. If you want to predict unseen chemistry, hold out whole substrates or ligands, using a custom split as in [§7.5](#75-train-from-python), and then score the held-out group with `scripts/evaluate.py`.


### 7.7 Benchmark on real data: the Perera et al. Suzuki dataset

The standard public benchmark for Suzuki-Miyaura yield prediction is the 5,760-reaction high-throughput screen of Perera et al. (*Science* 2018). It combines 7 electrophile/boron-partner pairings on a quinoline–indazole coupling with 12 ligand options (including none), 8 base options (including none) and 4 solvents, and records the yield of one product. Most of the variation in yield comes from the conditions, so a model that can't see the ligand, base and solvent can't do well on it.

`scripts/prepare_suzuki_dataset.py` downloads the paper's supplementary data and converts it to ReactionForge's CSV format. The data isn't redistributed with ReactionForge. It also writes the standard random splits from the [rxn_yields](https://github.com/rxn4chemistry/rxn_yields) repository: in each split file, the first 70 % of reactions are for training and the remaining 30 % for testing.

```bash
# 1. Download and convert split 0
python scripts/prepare_suzuki_dataset.py --split 0 --output_dir data/suzuki_split0

# 2. Train on the 70 % (10 % of it is held back for early stopping and calibration)
python scripts/train.py --data_path data/suzuki_split0/train.csv \
    --output_dir checkpoints/suzuki_split0 \
    --epochs 100 --patience 15 --no_temporal_memory --test_fraction 0 --val_fraction 0.1

# 3. Score on the untouched 30 %
python scripts/evaluate.py --checkpoint checkpoints/suzuki_split0/best_model.pt \
    --data_path data/suzuki_split0/test.csv --plot parity.png
```

Notes:

- `--no_temporal_memory` is used because this dataset has no meaningful reaction order. Without a `timestamp` column the model would otherwise read the row number as a timestamp.
- Add `--no_reagents` in step 2 to train the electrophile/product-only model, to measure how much the reagent encoding contributes.
- Training takes roughly an hour on a 4-core CPU. A GPU is much faster.
- To compare with published results, repeat for splits 0–9 and report the mean and standard deviation. A single split is noisy.

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

- They're **variances on the 0–1 yield scale**, not ± percentages. As a standard deviation in % points, use `sqrt(uncertainty) * 100`.
- The raw outputs of `model(...)` are **uncalibrated** (see §8.1). `scripts/predict.py`, `scripts/evaluate.py` and `src.inference.predict_dataframe` apply the calibration automatically.
- High *epistemic* uncertainty is the better signal for "run this experiment next". High *aleatoric* uncertainty means even more data won't make this reaction predictable.

### 8.1 Uncertainty calibration

The model's raw uncertainty is usually the wrong size. On the synthetic data in this guide the raw error bars were about 2.7× too wide: 99 % of true yields fell within ±1 standard deviation, where 68 % is expected. The *ranking* (which reactions are more uncertain than others) is still useful, but the numbers can't be read as error bars.

`scripts/train.py` fixes the size with **temperature scaling**: after training it finds the single factor `s` that best fits the predicted standard deviations to the validation-set errors, and saves it in the checkpoint as `uncertainty_scale`. Every prediction's standard deviation is then multiplied by `s` (variances by `s²`). On 1 000 held-out synthetic reactions this moved ±1 / ±2 std coverage from 99 % / 100 % to 67 % / 96 % (ideal 68 % / 95 %).

Limitations:

- **One factor for all reactions.** It corrects the overall size of the error bars, not their shape. If the model is overconfident on some kinds of reaction and underconfident on others, a single factor can't fix both.
- **It's fitted to the validation split's chemistry.** On substrates or conditions unlike the training data, the true errors can be larger than the calibrated error bars suggest. That's exactly when the epistemic uncertainty should be high, so treat high-epistemic predictions with extra caution.
- **Checkpoints trained before calibration was added** have no scale. They load with a scale of 1 (uncalibrated), and `evaluate.py` says so. Retrain to calibrate them.

To calibrate in your own code, use `fit_uncertainty_scale` (§10). If you call `model(...)` directly on a model from `load_model`, multiply `out['uncertainty']` by `model.uncertainty_scale ** 2`.

---

## 9. Configuring the Model

All settings are arguments to the `ReactionForge(...)` constructor. The YAML files in `configs/` are **not** read by any code.

| Argument | Default | What it controls | Guidance |
|---|---|---|---|
| `node_feat_dim` | 32 | Size of each atom's feature vector | **Must be 150** with `src/data/dataset.py`. The default of 32 won't work with real data. `scripts/train.py` sets it from the data automatically. |
| `edge_feat_dim` | 4 | Bond feature size | Stored but not used by the forward pass. Set it to 6 for consistency. |
| `condition_dim` | 10 | Length of the condition vector | Keep at 10 unless you change `extract_reaction_conditions` |
| `hidden_dim` | 128 | Width of internal layers | 64 (small data) to 256 (large data). Must be divisible by `num_heads`. |
| `num_wln_layers` | 3 | Message-passing depth (how far along a molecule information spreads) | 2–5 |
| `num_heads` | 8 | Attention heads | Must divide `hidden_dim` |
| `pooling_ratio` | 0.5 | Fraction of atoms kept by SAG pooling | Currently has **no effect**: the pooling layer is created but never called |
| `temporal_dim` | 64 | Size of the timestamp encoding | Must be even |
| `dropout` | 0.2 | Regularisation strength | Increase (0.3–0.5) if validation error rises while training error falls |
| `reagent_roles` | `()` | Which reagent columns to encode, e.g. `('boronic_acid', 'ligand', 'base', 'solvent')` | `scripts/train.py` sets it from the CSV's columns. Empty reproduces the original reactant/product-only model. |

Model size at the defaults with `node_feat_dim=150` is about 1.03 million parameters, or about 1.16 million with all four reagent roles.

**Turning features off.** To disable temporal memory, pass `--no_temporal_memory` to `scripts/train.py`, or don't pass `timestamps` to `model(...)`. Cross-attention can't be switched off without editing the code.

---

## 10. Evaluation and Uncertainty Tools

`scripts/evaluate.py` ([§7.3](#73-evaluate-on-held-out-data)) covers the usual case. For more detailed analysis, the functions below are standalone and work on NumPy arrays. Collect predictions first, using the 0–1 yield scale:

```python
import pandas as pd
from src.inference import load_model, predict_dataframe

model, use_timestamps = load_model("checkpoints/experiment_001/best_model.pt")
df = pd.read_csv("held_out.csv")
preds = predict_dataframe(model, df, use_timestamps=use_timestamps)
ok = preds["status"] == "ok"

y_true = df.loc[ok, "yield"].to_numpy() / 100
y_pred = preds.loc[ok, "predicted_yield"].to_numpy(dtype=float) / 100
epistemic = preds.loc[ok, "epistemic_var"].to_numpy(dtype=float)
aleatoric = preds.loc[ok, "aleatoric_var"].to_numpy(dtype=float)
```

Then:

| Function | Module | Purpose |
|---|---|---|
| `calculate_metrics(y_true, y_pred)` | `src.training.metrics` | Returns `{'r2', 'rmse', 'mae'}` on whatever scale you pass. Multiply RMSE/MAE by 100 for % points. |
| `expected_calibration_error(y_true, y_pred, uncertainty)` | `src.training.metrics` | Single calibration score. Lower is better. |
| `fit_uncertainty_scale(y_true, y_pred, variance)` | `src.utils.uncertainty` | The calibration factor `s` for standard deviations ([§8.1](#81-uncertainty-calibration)). Fit on validation data only. |
| `calibrate_uncertainty(y_true, y_pred, epistemic, aleatoric)` | `src.utils.uncertainty` | Applies that factor to epistemic and aleatoric *variances* |
| `coverage(y_true, y_pred, std, z)` | `src.utils.uncertainty` | Fraction of true values within `z` standard deviations. ≈ 0.68 (z=1) and 0.95 (z=2) when calibrated. |
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

All tests should pass:

```
28 passed
```

The same tests, plus a short train → predict → evaluate run on synthetic data and a documentation build, run automatically on GitHub for every pull request and every push to `main` (`.github/workflows/tests.yml`). A red ❌ next to a pull request means something broke. Open the **Details** link to see which step failed.

To build the documentation site locally (the same build Read the Docs runs):

```bash
pip install -r docs/requirements.txt
sphinx-build -W -b html docs docs/_build/html   # then open docs/_build/html/index.html
```

`-W` turns warnings into errors, as Read the Docs does. A malformed docstring in `src/` is enough to fail the build, so check this after editing docstrings.

If tests fail with `ModuleNotFoundError`, your installation or working directory is the problem (see [§12](#12-troubleshooting)).

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | Running from outside the repository root | `cd` to the root, or `export PYTHONPATH=/path/to/ReactionForge` |
| `ModuleNotFoundError: No module named 'torch'` / `'rdkit'` | Dependencies not installed in the active environment | Activate your venv or conda env, then re-run [§4.2](#42-install-with-pip-recommended) |
| `mat1 and mat2 shapes cannot be multiplied (…x150 and 154x128)` or `…and 32x128` | `node_feat_dim` doesn't match the data | Use `node_feat_dim=150` |
| `TypeError: forward() missing 2 required positional arguments` | Calling `model(batch_dict)` with the whole batch | Pass the parts: `model(batch['reactant'], batch['product'], batch['conditions'])`, or use `ReactionForgeTrainer` |
| `⚠️ Skipping N of M rows with invalid SMILES` | Unparseable SMILES in a reactant, product or reagent column | Not an error, but inspect those rows ([§6.3](#63-validate-your-file-before-training)) |
| Predictions ignore your ligand / base column | The model was trained without it (`Reagents encoded: none` in the training log), or the column name differs | Use the exact names in [§6.1](#61-file-format) and retrain |
| `IndexError: index … out of bounds` in `SinusoidalTemporalEncoding`, or `Timestamps must be in [0, 10000)` | A `timestamp` (or, with no `timestamp` column, the row number) ≥ 10 000 | Rescale timestamps below 10 000, or split the CSV into files of under 10 000 rows |
| `… has no 'model_config'` | Checkpoint written before `scripts/train.py` was fixed | Retrain, or rebuild `ReactionForge(...)` with the original arguments and call `load_state_dict` |
| `Input already has output column(s) …` | The input CSV is an earlier `predict.py` output | Drop the old prediction columns, or predict from the original file |
| `size mismatch for …` on `load_state_dict` | Constructor arguments differ from training | Recreate the model with identical arguments |
| `FileNotFoundError: checkpoints/reactionforge_best.pt` | No pretrained weights ship with the repo | Train your own ([§7.1](#71-train-from-the-command-line)) |
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

**License:** MIT. See [`LICENSE`](https://github.com/kroy3/ReactionForge/blob/main/LICENSE). You may use, modify, and redistribute the code, including commercially, provided the copyright notice is kept.

**Citation:** if you use ReactionForge in research, cite the accompanying preprint as given in the [README](https://github.com/kroy3/ReactionForge#-citation). The DOI there is currently a placeholder.

**Reporting problems:** open an issue at <https://github.com/kroy3/ReactionForge/issues> and include:

1. the exact command or script you ran,
2. the full error message,
3. the output of `python -c "import torch, torch_geometric, rdkit; print(torch.__version__, torch_geometric.__version__, rdkit.__version__)"`,
4. a few rows of your CSV, if the problem is data-related.
