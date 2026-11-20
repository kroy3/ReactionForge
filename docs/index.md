# ReactionForge

ReactionForge is a graph neural network that predicts the **yield of Suzuki-Miyaura cross-coupling reactions** from the reactant and product structures (SMILES) and the reaction conditions. Each prediction comes with a **calibrated uncertainty**, split into model uncertainty (*epistemic*) and data noise (*aleatoric*).

## Where to start

- **New to ReactionForge?** Read the [User Guide](USER_GUIDE.md). It covers installation, the input data format, training, prediction, evaluation, and troubleshooting.
- **Want to call it from Python?** See the [API Reference](api.md).

## In three commands

```bash
# 1. Train on your reactions (CSV: reactant_smiles, product_smiles, conditions, yield)
python scripts/train.py --data_path my_reactions.csv --output_dir checkpoints/exp1

# 2. Predict yields for new candidate reactions
python scripts/predict.py --checkpoint checkpoints/exp1/best_model.pt \
    --input_csv candidates.csv --output_csv predictions.csv

# 3. Score the model on held-out reactions
python scripts/evaluate.py --checkpoint checkpoints/exp1/best_model.pt \
    --data_path held_out.csv --plot parity.png
```

No pretrained weights are distributed, so step 1 is always needed. See [Project Status](USER_GUIDE.md#2-project-status-read-this-first) for what is and isn't implemented.

## Links

- Source code: <https://github.com/kroy3/ReactionForge>
- Issues: <https://github.com/kroy3/ReactionForge/issues>
- License: MIT

```{toctree}
:hidden:
:maxdepth: 2

USER_GUIDE
api
```
