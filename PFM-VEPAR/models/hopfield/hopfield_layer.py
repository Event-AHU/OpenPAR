"""
Modern Hopfield Layer Implementation

This module provides implementations of Modern Hopfield Layers for neural networks.
Based on the paper "Hopfield Networks is All You Need" and the AMTTrack implementation.

Author: Based on AMTTrack implementation
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from typing import Optional, Union


class HopfieldLayer(nn.Module):
    """
    Basic Modern Hopfield Layer implementation.
    
    This layer implements the modern Hopfield network mechanism for associative memory
    and feature enhancement in neural networks.
    
    Args:
        dim (int): Input feature dimension
        n_prototype (int): Number of prototypes (memory patterns). Default: 1000
        dropout (float): Dropout probability. Default: 0.1
        bias (bool): Whether to use bias in linear layers. Default: False
        temperature (Optional[float]): Temperature parameter. If None, uses 1/sqrt(dim)
    
    Shape:
        - Input: (batch_size, sequence_length, dim)
        - Output: (batch_size, sequence_length, dim)
    
    Examples:
        >>> hopfield = HopfieldLayer(dim=768, n_prototype=1000)
        >>> x = torch.randn(2, 100, 768)
        >>> output = hopfield(x)
        >>> print(output.shape)  # torch.Size([2, 100, 768])
    """
    
    def __init__(
        self, 
        dim: int, 
        n_prototype: int = 1000, 
        dropout: float = 0.1, 
        bias: bool = False,
        temperature: Optional[float] = None
    ):
        super().__init__()
        self.dim = dim
        self.n_prototype = n_prototype
        self.beta = temperature if temperature is not None else 1.0 / sqrt(dim)
        
        # Lookup matrix: maps input features to prototype space
        self.lookup_matrix = nn.Linear(dim, n_prototype, bias=bias)
        
        # Content matrix: maps from prototype space back to feature space
        self.content_matrix = nn.Linear(n_prototype, dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.lookup_matrix.weight)
        nn.init.xavier_uniform_(self.content_matrix.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Hopfield layer.
        
        Args:
            x (torch.Tensor): Input features [B, L, D]
            
        Returns:
            torch.Tensor: Enhanced features [B, L, D]
        """
        # Lookup phase: map to prototype space and compute attention
        lookup = torch.softmax(self.lookup_matrix(x) * self.beta, dim=-1)  # [B, L, P]
        
        # Retrieval phase: retrieve content from prototype space
        content = self.content_matrix(lookup)  # [B, L, D]
        
        return self.dropout(content)


class HopfieldLayerEnhanced(nn.Module):
    """
    Enhanced Hopfield Layer with LayerNorm and optional residual connection.
    
    This version includes layer normalization and residual connections for better
    training stability and performance.
    
    Args:
        dim (int): Input feature dimension
        n_prototype (int): Number of prototypes. Default: 1000
        dropout (float): Dropout probability. Default: 0.1
        use_layernorm (bool): Whether to use LayerNorm. Default: True
        residual (bool): Whether to use residual connection. Default: True
        temperature (Optional[float]): Temperature parameter. If None, uses 1/sqrt(dim)
    """
    
    def __init__(
        self, 
        dim: int, 
        n_prototype: int = 1000, 
        dropout: float = 0.1,
        use_layernorm: bool = True,
        residual: bool = True,
        temperature: Optional[float] = None
    ):
        super().__init__()
        self.dim = dim
        self.n_prototype = n_prototype
        self.beta = temperature if temperature is not None else 1.0 / sqrt(dim)
        self.residual = residual
        
        self.lookup_matrix = nn.Linear(dim, n_prototype, bias=False)
        
        if use_layernorm:
            self.content = nn.Sequential(
                nn.Linear(n_prototype, dim, bias=False),
                nn.LayerNorm(dim),
            )
        else:
            self.content = nn.Linear(n_prototype, dim, bias=False)
            
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.lookup_matrix.weight)
        if isinstance(self.content, nn.Sequential):
            nn.init.xavier_uniform_(self.content[0].weight)
        else:
            nn.init.xavier_uniform_(self.content.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connection."""
        # Save residual
        residual = x if self.residual else 0
        
        # Hopfield operation
        lookup = torch.softmax(self.lookup_matrix(x) * self.beta, dim=-1)
        content = self.content(lookup)
        output = self.dropout(content)
        
        # Residual connection
        return output + residual


class ConditionalHopfieldLayer(nn.Module):
    """
    Conditional Hopfield Layer that can be applied selectively based on layer index.
    
    This is useful in transformer architectures where you want to apply Hopfield
    enhancement only in certain layers (e.g., first 8 layers).
    
    Args:
        dim (int): Input feature dimension
        n_prototype (int): Number of prototypes. Default: 1000
        dropout (float): Dropout probability. Default: 0.1
        max_layer (int): Maximum layer index to apply Hopfield. Default: 8
        use_layernorm (bool): Whether to use LayerNorm. Default: True
    """
    
    def __init__(
        self, 
        dim: int, 
        n_prototype: int = 1000, 
        dropout: float = 0.1,
        max_layer: int = 8,
        use_layernorm: bool = True
    ):
        super().__init__()
        self.max_layer = max_layer
        self.hopfield = HopfieldLayerEnhanced(
            dim=dim,
            n_prototype=n_prototype,
            dropout=dropout,
            use_layernorm=use_layernorm,
            residual=True
        )
        
    def forward(self, x: torch.Tensor, layer_idx: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass with conditional application.
        
        Args:
            x (torch.Tensor): Input features
            layer_idx (Optional[int]): Current layer index. If None or >= max_layer,
                                     returns input unchanged.
        
        Returns:
            torch.Tensor: Enhanced features if layer_idx < max_layer, else input
        """
        if layer_idx is not None and layer_idx < self.max_layer:
            return self.hopfield(x)
        else:
            return x


class MultiScaleHopfieldLayer(nn.Module):
    """
    Multi-scale Hopfield Layer with different prototype numbers for different scales.
    
    This layer uses multiple Hopfield layers with different prototype numbers
    to capture features at different scales.
    
    Args:
        dim (int): Input feature dimension
        prototype_scales (list): List of prototype numbers for different scales
        dropout (float): Dropout probability. Default: 0.1
        fusion_method (str): Method to fuse multi-scale features. 
                           Options: 'concat', 'add', 'weighted'. Default: 'add'
    """
    
    def __init__(
        self, 
        dim: int, 
        prototype_scales: list = [500, 1000, 2000], 
        dropout: float = 0.1,
        fusion_method: str = 'add'
    ):
        super().__init__()
        self.fusion_method = fusion_method
        
        # Create multiple Hopfield layers with different scales
        self.hopfield_layers = nn.ModuleList([
            HopfieldLayer(dim, n_prototype, dropout) 
            for n_prototype in prototype_scales
        ])
        
        if fusion_method == 'concat':
            self.fusion_proj = nn.Linear(dim * len(prototype_scales), dim)
        elif fusion_method == 'weighted':
            self.fusion_weights = nn.Parameter(torch.ones(len(prototype_scales)))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-scale processing."""
        # Apply each Hopfield layer
        outputs = [layer(x) for layer in self.hopfield_layers]
        
        # Fuse outputs
        if self.fusion_method == 'concat':
            fused = torch.cat(outputs, dim=-1)
            return self.fusion_proj(fused)
        elif self.fusion_method == 'add':
            return sum(outputs)
        elif self.fusion_method == 'weighted':
            weights = torch.softmax(self.fusion_weights, dim=0)
            return sum(w * out for w, out in zip(weights, outputs))
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")


# Utility functions for integration with existing architectures

def add_hopfield_to_transformer_block(
    transformer_block: nn.Module,
    dim: int,
    n_prototype: int = 1000,
    position: str = 'before_attention'
) -> nn.Module:
    """
    Add Hopfield layer to an existing transformer block.
    
    Args:
        transformer_block: Existing transformer block
        dim: Feature dimension
        n_prototype: Number of prototypes
        position: Where to add Hopfield layer ('before_attention' or 'after_attention')
    
    Returns:
        Modified transformer block with Hopfield layer
    """
    class TransformerBlockWithHopfield(nn.Module):
        def __init__(self, original_block, hopfield_layer, position):
            super().__init__()
            self.original_block = original_block
            self.hopfield = hopfield_layer
            self.position = position
            
        def forward(self, x, *args, **kwargs):
            if self.position == 'before_attention':
                x = self.hopfield(x)
                return self.original_block(x, *args, **kwargs)
            else:  # after_attention
                x = self.original_block(x, *args, **kwargs)
                return self.hopfield(x)
    
    hopfield_layer = HopfieldLayerEnhanced(dim, n_prototype)
    return TransformerBlockWithHopfield(transformer_block, hopfield_layer, position)


def create_hopfield_enhanced_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    n_prototype: int = 1000,
    dropout: float = 0.1
) -> nn.Module:
    """
    Create an MLP with Hopfield enhancement.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        n_prototype: Number of prototypes for Hopfield layer
        dropout: Dropout probability
    
    Returns:
        MLP with Hopfield enhancement
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        HopfieldLayerEnhanced(hidden_dim, n_prototype, dropout),
        nn.Linear(hidden_dim, output_dim)
    )


# Example usage and test functions


if __name__ == "__main__":
 

# -----------------------------
# External memory-augmented Hopfield (Query from input; Key/Value from external memory)
# -----------------------------

import math

class ExternalMemoryHopfield(nn.Module):
    """
    Memory-augmented Hopfield where queries come from the current features and
    keys/values are provided by an external memory bank.

    Args:
        dim: feature dimension of queries and memory entries
        dropout: dropout applied to the output
        temperature: attention temperature; default 1/sqrt(dim)
        proj_qkv: whether to apply learnable linear projections to q/k/v
        freeze_memory: whether external memory is trainable
    """

    def __init__(self, dim: int, dropout: float = 0.1, temperature: Optional[float] = None,
                 proj_qkv: bool = True, freeze_memory: bool = True):
        super().__init__()
        self.dim = dim
        self.beta = temperature if temperature is not None else 1.0 / math.sqrt(dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_qkv = proj_qkv

        if proj_qkv:
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)
        else:
            self.q_proj = self.k_proj = self.v_proj = nn.Identity()

        # External memory parameter placeholder. Shape expected: [1, M, D]
        self.register_parameter('memory', None)
        self.freeze_memory = freeze_memory

    def set_memory(self, mem: torch.Tensor):
        """
        Set external memory entries.

        mem: [M, D] or [1, M, D]
        """
        if mem.dim() == 2:
            mem = mem.unsqueeze(0)
        param = nn.Parameter(mem, requires_grad=not self.freeze_memory)
        self.memory = param

    @torch.no_grad()
    def load_memory(self, path: str):
        """Load memory entries from .pt/.pth/.npy/.pkl file."""
        import numpy as np, pickle
        if path.endswith('.pt') or path.endswith('.pth'):
            mem = torch.load(path, map_location='cpu')
        elif path.endswith('.npy'):
            mem = torch.from_numpy(np.load(path))
        elif path.endswith('.pkl') or path.endswith('.pickle'):
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            mem = torch.as_tensor(obj)
        else:
            raise ValueError(f'Unsupported memory file: {path}')
        self.set_memory(mem.float())

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取注意力权重，用于跨模态一致性分析
        x: [B, N, D]
        returns: [B, N, M] attention weights
        """
        assert self.memory is not None, "ExternalMemoryHopfield: call set_memory/load_memory before forward"
        B, N, D = x.shape
        mem = self.memory  # [1, M, D]

        q = self.q_proj(x)                    # [B, N, D]
        k = self.k_proj(mem)                  # [1, M, D]

        attn_logits = self.beta * torch.matmul(q, k.transpose(-1, -2))  # [B, N, M]
        attn_weights = torch.softmax(attn_logits, dim=-1)
        return attn_weights

    def forward_with_attention(self, x: torch.Tensor) -> tuple:
        """
        返回检索结果和注意力权重
        x: [B, N, D]
        returns: (retrieved_features, attention_weights)
        """
        assert self.memory is not None, "ExternalMemoryHopfield: call set_memory/load_memory before forward"
        B, N, D = x.shape
        mem = self.memory  # [1, M, D]

        q = self.q_proj(x)                    # [B, N, D]
        k = self.k_proj(mem)                  # [1, M, D]
        v = self.v_proj(mem)                  # [1, M, D]

        attn_logits = self.beta * torch.matmul(q, k.transpose(-1, -2))  # [B, N, M]
        attn_weights = torch.softmax(attn_logits, dim=-1)
        out = torch.matmul(attn_weights, v.expand(B, -1, -1))  # [B, N, D]
        
        return self.dropout(out), attn_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D]
        returns: [B, N, D]
        """
        out, _ = self.forward_with_attention(x)
        return out