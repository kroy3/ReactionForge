"""
Training Loop with Evidential Loss
===================================
Complete training orchestration for ReactionForge.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


class ReactionForgeTrainer:
    """Trainer class for ReactionForge model"""
    
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            output = self.model(batch)
            loss = output['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating'):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                output = self.model(batch)
                loss = output['loss']
                
                total_loss += loss.item()
                predictions.extend(output['yield_mean'].cpu().numpy())
                targets.extend(batch['yield'].cpu().numpy())
        
        # Calculate metrics
        import numpy as np
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        r2 = 1 - np.sum((targets - predictions)**2) / np.sum((targets - targets.mean())**2)
        rmse = np.sqrt(np.mean((targets - predictions)**2))
        mae = np.mean(np.abs(targets - predictions))
        
        return {
            'loss': total_loss / len(val_loader),
            'r2': r2,
            'rmse': rmse,
            'mae': mae
        }
