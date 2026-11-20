# API Reference

Generated from the docstrings in `src/`. Import everything from the repository root (for example `from src.inference import load_model`). See the [User Guide](USER_GUIDE.md) for worked examples.

## Inference

Load a trained checkpoint and predict a table of reactions. These are the functions behind `scripts/predict.py` and `scripts/evaluate.py`.

```{eval-rst}
.. automodule:: src.inference
```

## Model

```{eval-rst}
.. automodule:: src.models.reactionforge
   :members: ReactionForge, evidential_loss
```

## Data

```{eval-rst}
.. automodule:: src.data.dataset
   :members: SuzukiMiyauraDataset, REAGENT_ROLES, reagent_inputs, invalid_smiles_columns, detect_reagent_roles, suzuki_product, smiles_to_graph, extract_reaction_conditions, generate_synthetic_dataset
```

```{eval-rst}
.. automodule:: src.data.augmentation
```

## Training

```{eval-rst}
.. automodule:: src.training.trainer
```

```{eval-rst}
.. automodule:: src.training.callbacks
```

```{eval-rst}
.. automodule:: src.training.metrics
```

## Uncertainty and plotting

```{eval-rst}
.. automodule:: src.utils.uncertainty
```

```{eval-rst}
.. automodule:: src.utils.visualization
```
