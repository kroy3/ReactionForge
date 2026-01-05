"""
Cross-Attention Mechanisms
===========================
Multi-head attention for reactant-product comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention between reactants and products"""
    
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        """
        Cross-attention from query to key/value
        
        Args:
            query: Reactant features [batch, seq_len_q, d_model]
            key: Product features [batch, seq_len_k, d_model]
            value: Product features [batch, seq_len_v, d_model]
        """
        batch_size = query.size(0)
        residual = query
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attended)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output


class SelfAttention(nn.Module):
    """Self-attention for long-range dependencies"""
    
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttention(d_model, num_heads, dropout)
        
    def forward(self, x, mask=None):
        """Self-attention by using x as query, key, and value"""
        return self.cross_attn(x, x, x, mask)
