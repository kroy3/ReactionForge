# ReactionForge: Temporal Graph Network for Reaction Yield Prediction

<p align="center">
  <strong>Graph neural network for Suzuki-Miyaura yield prediction with calibrated uncertainty</strong>
</p>

<p align="center">
  <a href="https://chemrxiv.org/engage/chemrxiv/article-details/XXXXXXXXX"><img src="https://img.shields.io/badge/ChemRxiv-Preprint-orange" alt="ChemRxiv"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"></a>
</p>

---

## 🔬 Overview

**ReactionForge** predicts the yield of Suzuki-Miyaura cross-coupling reactions, with an uncertainty estimate for every prediction. Each reaction is described by molecular structures (SMILES) of its components:

- **Electrophile** (aryl halide or triflate) and **product**
- **Boron coupling partner**, **ligand**, **base** and **solvent**, each optional
- **Temperature**, **time** and **catalyst loading**, also optional

How the model works:

1. **Molecular graph encoding.** Every component is a graph of atoms and bonds, encoded with shared Weisfeiler-Lehman message-passing layers. Ligands, bases and solvents are read as *structures*, not names from a fixed list.
2. **Reactant-product cross-attention** relates the electrophile to the product.
3. **Evidential uncertainty.** The output layer predicts a Normal-Inverse-Gamma distribution, giving epistemic (model) and aleatoric (data) uncertainty in one forward pass. After training, the uncertainty is calibrated on a validation set.
4. **Optional temporal memory.** A GRU can carry state across a sequence of reactions when a `timestamp` column is supplied. It has not been evaluated on a dataset with meaningful reaction order.

The model also has a selectivity head and a SAGPool layer, but neither is trained or used yet. See the [project status](https://reactionforge.readthedocs.io/en/latest/USER_GUIDE.html#2-project-status-read-this-first) for what is and isn't implemented.

### Benchmark

Measured on the 5,760-reaction high-throughput Suzuki-Miyaura dataset of Perera et al. (*Science* 2018). Standard random split 0 from rxn_yields: trained on 70% of the reactions and scored on the other 30% (1,728 reactions the model never saw).

| Model | R² | RMSE (yield %) | MAE (yield %) | Within ±1σ / ±2σ |
|---|---|---|---|---|
| ReactionForge (boron partner, ligand, base, solvent) | **0.835** | **11.3** | **7.7** | 74% / 95% |
| ReactionForge, electrophile + product only | 0.177 | 25.2 | 20.9 | 63% / 98% |

- **Run conditions:** a single split and seed on CPU, stopped at the 100-epoch limit. Means over all 10 standard splits, as reported in the literature, haven't been run yet.
- **Calibration:** the uncertainty is calibrated on a validation subset; ideal coverage is 68% / 95%.
- **Weak spot:** the model is least reliable on failed reactions in the reversed coupling orientation. See the [benchmark notes](https://reactionforge.readthedocs.io/en/latest/USER_GUIDE.html#77-benchmark-on-real-data-the-perera-et-al-suzuki-dataset).

Reproduce it with the commands in [Reproducing the Benchmark](#-reproducing-the-benchmark).

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ReactionForge.git
cd ReactionForge

# Create conda environment
conda create -n reactionforge python=3.10
conda activate reactionforge

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric (CPU)
pip install torch-geometric torch-scatter torch-sparse

# For GPU support (CUDA 11.8)
pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```

### Quick Prediction

No pretrained weights ship with this repository, so first train a model (see [Training Your Own Model](#-training-your-own-model)). Then:

```bash
# Score a CSV of candidate reactions (reactant_smiles, product_smiles, conditions)
python scripts/predict.py \
    --checkpoint checkpoints/experiment_001/best_model.pt \
    --input_csv candidates.csv \
    --output_csv predictions.csv
```

Or from Python:

```python
import torch
import pandas as pd
from torch_geometric.data import Batch
from src.inference import load_model
from src.data.dataset import smiles_to_graph, extract_reaction_conditions

model, _ = load_model('checkpoints/experiment_001/best_model.pt')

reactant = Batch.from_data_list([smiles_to_graph('c1ccc(Br)cc1')])        # Bromobenzene
product = Batch.from_data_list([smiles_to_graph('c1ccc(-c2ccccc2)cc1')])   # Biphenyl
conditions = extract_reaction_conditions(
    pd.Series({'temperature': 90, 'time': 12, 'catalyst_loading': 5})).unsqueeze(0)

with torch.no_grad():
    output = model(reactant, product, conditions)

print(f"Predicted yield: {output['yield_mean'].item()*100:.1f}%")
std = output['uncertainty'].sqrt().item() * model.uncertainty_scale  # calibrated in train.py
print(f"Uncertainty (std): ±{std*100:.1f}%")
```

See the [User Guide](docs/USER_GUIDE.md#81-uncertainty-calibration) for how the uncertainty is calibrated and how to interpret it.

---

## 📂 Repository Structure

```
ReactionForge/
├── src/
│   ├── models/
│   │   ├── reactionforge.py      # Core TGN architecture
│   │   ├── wln_layers.py          # Weisfeiler-Lehman networks
│   │   └── attention.py           # Cross-attention modules
│   ├── data/
│   │   ├── dataset.py             # PyG dataset classes
│   │   ├── featurization.py       # Molecular feature extraction
│   │   └── augmentation.py        # Data augmentation strategies
│   ├── training/
│   │   ├── trainer.py             # Training loop with evidential loss
│   │   ├── callbacks.py           # Early stopping, checkpointing
│   │   └── metrics.py             # Evaluation metrics
│   ├── utils/
│   │   ├── visualization.py       # Plotting utilities
│   │   └── uncertainty.py         # Uncertainty calibration
│   └── inference.py               # Load checkpoints, batch prediction
├── scripts/
│   ├── prepare_suzuki_dataset.py  # Download/convert the Perera et al. benchmark
│   ├── train.py                   # Main training script
│   ├── evaluate.py                # Evaluation on test set
│   ├── hyperopt.py                # Hyperparameter search (placeholder, not implemented)
│   └── predict.py                 # Batch prediction
├── notebooks/
│   ├── 01_quickstart.ipynb        # Getting started tutorial
│   ├── 02_training.ipynb          # Training walkthrough
│   ├── 03_analysis.ipynb          # Result analysis
│   └── 04_uncertainty.ipynb       # Uncertainty quantification
├── configs/
│   ├── default.yaml               # Default hyperparameters
│   ├── ablation.yaml              # Ablation study configs
│   └── transfer_learning.yaml     # Transfer learning setup
├── tests/
│   ├── test_model.py              # Unit tests for model
│   ├── test_data.py               # Data processing tests
│   └── test_training.py           # Training pipeline tests
├── figures/                       # Paper figures
├── checkpoints/                   # Pretrained model weights
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment
├── README.md                      # This file
└── LICENSE                        # MIT License
```

---

## 🎓 Training Your Own Model

### Basic Training

```bash
# Train on Suzuki-Miyaura dataset
python scripts/train.py \
    --data_path data/suzuki_reactions.csv \
    --output_dir checkpoints/experiment_001 \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --hidden_dim 128 \
    --num_wln_layers 3
```

The best model is saved to `<output_dir>/best_model.pt`. All settings are command-line options: run `python scripts/train.py --help`, or see the [User Guide](docs/USER_GUIDE.md#71-train-from-the-command-line). The CSV columns the model reads, including ligand, base and solvent, are listed in the [data format section](docs/USER_GUIDE.md#61-file-format).

The YAML files in `configs/` are not read by any code yet, and `scripts/hyperopt.py` is a placeholder that does not train models.

---

## 📊 Reproducing the Benchmark

The benchmark uses the 5,760-reaction high-throughput Suzuki-Miyaura dataset of Perera et al. (*Science* 2018), with the standard random splits from [rxn_yields](https://github.com/rxn4chemistry/rxn_yields). The data isn't redistributed here: the preparation script downloads it.

```bash
# 1. Download and convert split 0 (70% train / 30% test)
python scripts/prepare_suzuki_dataset.py --split 0 --output_dir data/suzuki_split0

# 2. Train on the 70%, holding out 10% of it for early stopping and calibration
python scripts/train.py --data_path data/suzuki_split0/train.csv \
    --output_dir checkpoints/suzuki_split0 \
    --epochs 100 --patience 15 --no_temporal_memory --test_fraction 0 --val_fraction 0.1

# 3. Score on the untouched 30% test set
python scripts/evaluate.py --checkpoint checkpoints/suzuki_split0/best_model.pt \
    --data_path data/suzuki_split0/test.csv --plot parity.png
```

Add `--no_reagents` to step 2 to train the reactant/product-only model, the ablation in the table above. Repeat with `--split 1` to `9` for the mean and spread over splits.

### Evaluating a Trained Model

```bash
# Score a model on held-out reactions (needs a yield column)
python scripts/evaluate.py \
    --checkpoint checkpoints/experiment_001/best_model.pt \
    --data_path held_out.csv \
    --output_csv evaluation.csv \
    --plot parity.png
```

Reports R², RMSE, MAE, calibration error and uncertainty coverage. For out-of-distribution testing (for example leave-one-ligand-out), build the held-out CSV yourself so it contains only ligands or substrates absent from the training CSV.

---

## 📖 Documentation

📘 **Start with the [User Guide](docs/USER_GUIDE.md)**: installation, data format, a tested train-and-predict example, output interpretation, and troubleshooting.

Full documentation is available at **[reactionforge.readthedocs.io](https://reactionforge.readthedocs.io)**

### Key Topics

- [Project Status](https://reactionforge.readthedocs.io/en/latest/USER_GUIDE.html#2-project-status-read-this-first) - What is and isn't implemented
- [Data Preparation](https://reactionforge.readthedocs.io/en/latest/USER_GUIDE.html#6-preparing-your-data) - How to format your own datasets
- [Training, Prediction and Evaluation](https://reactionforge.readthedocs.io/en/latest/USER_GUIDE.html#7-quick-start-train-and-predict) - Command-line workflow
- [Uncertainty Calibration](https://reactionforge.readthedocs.io/en/latest/USER_GUIDE.html#81-uncertainty-calibration) - How to read the error bars
- [API Reference](https://reactionforge.readthedocs.io/en/latest/api.html) - Python API generated from the docstrings
- [Troubleshooting](https://reactionforge.readthedocs.io/en/latest/USER_GUIDE.html#12-troubleshooting) - Common errors and fixes

---

## 🤝 Citation

If you use ReactionForge in your research, please cite our paper:

```bibtex
@article{roy2025reactionforge,
  title={ReactionForge: Temporal Graph Networks Surpass State-of-the-Art in Suzuki-Miyaura Yield Prediction},
  author={Roy, Kushal Raj},
  journal={ChemRxiv},
  year={2025},
  doi={10.XXXX/chemrxiv.XXXXXXX}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Perera et al.** (*Science* 2018) for the Suzuki-Miyaura high-throughput dataset, and **Schwaller et al.** (rxn_yields) for the standard splits
- **Chemprop v2.0** (Heid et al., 2024) for evidential deep learning implementation
- **PyTorch Geometric** team for excellent graph learning tools
- **University of Houston** Department of Biology & Biochemistry

---

## 💬 Contact

**Kushal Raj Roy**  
University of Houston  
📧 kroy@uh.edu  
🔗 [LinkedIn](https://linkedin.com/in/kushalrajroy) | [Google Scholar](https://scholar.google.com)

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=kroy3/ReactionForge&type=Date)](https://star-history.com/#kroy3/ReactionForge&Date)

---

**Built with ❤️ for the chemistry community**
