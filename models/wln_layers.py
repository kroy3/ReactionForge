"""
Weisfeiler-Lehman Network Layers
=================================
Edge-type-specific message passing for chemical graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class WLNLayer(MessagePassing):
    """WLN layer with edge-type-specific aggregation"""
    
    def __init__(self, in_channels, out_channels, num_edge_types=4):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Edge-type-specific MLPs
        self.edge_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.LayerNorm(out_channels)
            )
            for _ in range(num_edge_types)
        ])
        
        self.self_loop = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr):
        """Forward pass with WLN message passing"""
        # Add self-loops
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
        
        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Add self-loop contribution
        out = out + self.self_loop(x)
        
        return out
    
    def message(self, x_i, x_j, edge_attr):
        """Construct edge-type-specific messages"""
        # Concatenate source and target features
        x_cat = torch.cat([x_i, x_j], dim=-1)
        
        # Get edge types
        edge_types = edge_attr.argmax(dim=-1) if edge_attr.dim() > 1 else edge_attr
        
        # Apply edge-type-specific networks
        messages = torch.zeros(x_cat.size(0), self.out_channels, device=x_cat.device)
        for edge_type in range(len(self.edge_networks)):
            mask = (edge_types == edge_type)
            if mask.sum() > 0:
                messages[mask] = self.edge_networks[edge_type](x_cat[mask])
        
        return messages
