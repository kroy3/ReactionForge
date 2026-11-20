"""
Training Loop with Evidential Loss
===================================
Complete training orchestration for ReactionForge.

Batches are the dicts produced by ``SuzukiMiyauraDataset`` and collated by
``torch_geometric.loader.DataLoader``:

    {'reactant': Batch, 'product': Batch, 'conditions': (B, C),
     'yield': (B, 1) on a 0-1 scale, 'selectivity': (B, 1), 'timestamp': (B, 1)}
"""

import torch
from tqdm import tqdm

from src.data.dataset import reagent_inputs
from src.models.reactionforge import evidential_loss
from src.training.metrics import calculate_metrics


class ReactionForgeTrainer:
    """Trainer class for ReactionForge model"""

    def __init__(self, model, optimizer, device='cuda', lambda_reg=0.01,
                 use_timestamps=True, max_grad_norm=1.0):
        """
        Args:
            model: ReactionForge instance
            optimizer: torch optimizer over ``model.parameters()``
            device: device to train on
            lambda_reg: weight of the evidential regulariser
            use_timestamps: pass reaction timestamps to the model (enables
                temporal memory); set False to train without it
            max_grad_norm: gradient clipping threshold
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lambda_reg = lambda_reg
        self.use_timestamps = use_timestamps
        self.max_grad_norm = max_grad_norm
        self.model.to(device)

    def _forward(self, batch):
        """Move a batch to the device, run the model, and compute the loss."""
        batch = {k: v.to(self.device) if hasattr(v, 'to') else v
                 for k, v in batch.items()}

        timestamps = batch['timestamp'] if self.use_timestamps else None
        output = self.model(batch['reactant'], batch['product'],
                            batch['conditions'], timestamps,
                            reagents=reagent_inputs(batch, self.model.reagent_roles))

        gamma, nu, alpha, beta = output['yield_params']
        target = batch['yield'].view(-1, 1)
        loss = evidential_loss(gamma, nu, alpha, beta, target, self.lambda_reg)
        return output, target, loss

    def train_epoch(self, train_loader):
        """Train for one epoch. Returns the mean training loss."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            _, _, loss = self._forward(batch)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           max_norm=self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(train_loader)

    def collect_predictions(self, loader):
        """
        Run the model over ``loader`` without gradients.

        Returns:
            (y_true, y_pred, variance) NumPy arrays on the 0-1 yield scale, where
            variance is the total predictive variance (epistemic + aleatoric).
        """
        self.model.eval()
        y_true, y_pred, variance = [], [], []
        with torch.no_grad():
            for batch in loader:
                output, target, _ = self._forward(batch)
                y_true.append(target.view(-1).cpu())
                y_pred.append(output['yield_mean'].view(-1).cpu())
                variance.append(output['uncertainty'].view(-1).cpu())
        return tuple(torch.cat(t).numpy() for t in (y_true, y_pred, variance))

    def evaluate(self, val_loader):
        """
        Evaluate on a validation or test set.

        Returns:
            dict with 'loss', 'r2', and 'rmse' / 'mae' in yield percentage points.
        """
        self.model.eval()
        total_loss = 0.0
        predictions, targets = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating'):
                output, target, loss = self._forward(batch)
                total_loss += loss.item()
                predictions.append(output['yield_mean'].view(-1).cpu())
                targets.append(target.view(-1).cpu())

        # Yields are stored on a 0-1 scale; report errors in percentage points.
        predictions = torch.cat(predictions).numpy() * 100
        targets = torch.cat(targets).numpy() * 100
        metrics = calculate_metrics(targets, predictions)

        return {
            'loss': total_loss / len(val_loader),
            'r2': float(metrics['r2']),
            'rmse': float(metrics['rmse']),
            'mae': float(metrics['mae']),
        }
