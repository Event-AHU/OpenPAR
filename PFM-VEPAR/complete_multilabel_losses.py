"""
Complete Multi-Label Classification Loss Functions
=================================================

A comprehensive collection of loss functions specifically designed for multi-label 
classification tasks, with special focus on handling class imbalance in scenarios 
like pedestrian attribute recognition.

Features:
- Multiple Loss Types: Standard BCE, Weighted BCE, Adaptive BCE, and Scaled BCE
- Automatic Class Balancing: Built-in mechanisms to handle imbalanced datasets
- Flexible Configuration: Easy-to-use configuration system with presets
- Label Smoothing: Optional label smoothing for better generalization
- Multiple Weighting Methods: TIP2020, Focal Loss, and Inverse Frequency weighting
- PyTorch Integration: Seamless integration with PyTorch training loops

Author: Based on Rethinking_of_PAR project
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any
from dataclasses import dataclass, field
import warnings


# ============================================================================
# Utility Functions
# ============================================================================

def compute_class_weights(labels: np.ndarray, method: str = 'inverse_freq') -> np.ndarray:
    """
    Compute class weights for handling imbalanced multi-label data.
    
    Args:
        labels: Binary label matrix of shape (num_samples, num_classes)
        method: Weight computation method
            - 'inverse_freq': 1 / frequency
            - 'sqrt_inverse_freq': 1 / sqrt(frequency)  
            - 'balanced': sklearn-style balanced weights
            - 'tip20': TIP2020 paper method (exponential weighting)
    
    Returns:
        Class weights array of shape (num_classes,)
    """
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    # Compute positive class ratios
    pos_ratios = labels.mean(axis=0)
    
    if method == 'inverse_freq':
        weights = 1.0 / (pos_ratios + 1e-8)
    elif method == 'sqrt_inverse_freq':
        weights = 1.0 / (np.sqrt(pos_ratios) + 1e-8)
    elif method == 'balanced':
        weights = len(labels) / (2.0 * np.sum(labels, axis=0) + 1e-8)
    elif method == 'tip20':
        # TIP2020 method: return ratios for later exponential weighting
        weights = pos_ratios
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    return weights


def ratio2weight_tip20(targets: torch.Tensor, ratios: torch.Tensor) -> torch.Tensor:
    """
    Convert class ratios to sample weights using TIP2020 method.
    
    This method assigns higher weights to rare classes using exponential scaling:
    - For positive samples: weight = exp(1 - ratio)
    - For negative samples: weight = exp(ratio)
    
    Args:
        targets: Binary targets of shape (batch_size, num_classes)
        ratios: Positive class ratios of shape (num_classes,)
    
    Returns:
        Sample weights of shape (batch_size, num_classes)
    """
    ratios = ratios.type_as(targets)
    
    # Compute weights using exponential scaling
    pos_weights = targets * (1 - ratios)      # Higher weight for rare positive classes
    neg_weights = (1 - targets) * ratios      # Higher weight for rare negative classes
    weights = torch.exp(neg_weights + pos_weights)
    
    # Handle edge cases (targets > 1 should have zero weight)
    weights[targets > 1] = 0.0
    
    return weights


# ============================================================================
# Loss Functions
# ============================================================================

class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for multi-label classification.
    
    This loss function addresses class imbalance by applying different weights
    to positive and negative samples based on class frequencies.
    
    Args:
        pos_weights: Positive class weights of shape (num_classes,)
        reduction: Reduction method ('mean', 'sum', 'none')
        label_smoothing: Label smoothing factor (0.0 to 1.0)
        size_sum: If True, sum over classes then mean over batch
    
    Example:
        >>> pos_weights = torch.tensor([2.0, 1.5, 3.0])  # Higher weights for rare classes
        >>> loss_fn = WeightedBCELoss(pos_weights=pos_weights, label_smoothing=0.1)
        >>> loss, loss_matrix = loss_fn(logits, targets)
    """
    
    def __init__(self, 
                 pos_weights: Optional[torch.Tensor] = None,
                 reduction: str = 'mean',
                 label_smoothing: float = 0.0,
                 size_sum: bool = True):
        super().__init__()
        
        self.register_buffer('pos_weights', pos_weights)
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.size_sum = size_sum
        
        if label_smoothing < 0.0 or label_smoothing > 1.0:
            raise ValueError(f"Label smoothing must be in [0, 1], got {label_smoothing}")
    
    def forward(self, 
                logits: torch.Tensor, 
                targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            logits: Predicted logits of shape (batch_size, num_classes)
            targets: Binary targets of shape (batch_size, num_classes)
        
        Returns:
            Tuple of (total_loss, per_sample_losses)
        """
        # Handle list input (for compatibility)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = ((1 - self.label_smoothing) * targets + 
                      self.label_smoothing * (1 - targets))
        
        # Compute BCE loss without reduction
        loss_matrix = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weights, reduction='none'
        )
        
        # Apply reduction
        if self.size_sum:
            # Sum over classes, then mean over batch
            loss = loss_matrix.sum(dim=1).mean()
        else:
            if self.reduction == 'mean':
                loss = loss_matrix.mean()
            elif self.reduction == 'sum':
                loss = loss_matrix.sum()
            elif self.reduction == 'none':
                loss = loss_matrix
            else:
                raise ValueError(f"Unknown reduction: {self.reduction}")
        
        return loss, loss_matrix


class AdaptiveWeightedBCELoss(nn.Module):
    """
    Adaptive Weighted BCE Loss with TIP2020-style sample weighting.
    
    This loss automatically computes sample weights based on class frequencies
    and current batch statistics, providing dynamic class balancing.
    
    Args:
        class_ratios: Positive class ratios from training data
        reduction: Reduction method ('mean', 'sum', 'none')  
        label_smoothing: Label smoothing factor
        size_sum: If True, sum over classes then mean over batch
        weight_method: Method for computing weights ('tip20', 'focal')
    
    Example:
        >>> class_ratios = np.array([0.1, 0.3, 0.05, 0.8])  # Class frequencies
        >>> loss_fn = AdaptiveWeightedBCELoss(class_ratios=class_ratios, 
        ...                                   weight_method='tip20',
        ...                                   label_smoothing=0.1)
        >>> loss, loss_matrix = loss_fn(logits, targets)
    """
    
    def __init__(self,
                 class_ratios: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 reduction: str = 'mean',
                 label_smoothing: float = 0.0,
                 size_sum: bool = True,
                 weight_method: str = 'tip20'):
        super().__init__()
        
        if class_ratios is not None:
            if isinstance(class_ratios, np.ndarray):
                class_ratios = torch.from_numpy(class_ratios).float()
            self.register_buffer('class_ratios', class_ratios)
        else:
            self.class_ratios = None
            
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.size_sum = size_sum
        self.weight_method = weight_method
    
    def forward(self, 
                logits: torch.Tensor, 
                targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with adaptive weighting."""
        
        # Handle list input
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = ((1 - self.label_smoothing) * targets + 
                      self.label_smoothing * (1 - targets))
        
        # Compute base BCE loss
        loss_matrix = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Apply adaptive weighting if class ratios are provided
        if self.class_ratios is not None:
            if self.weight_method == 'tip20':
                weights = ratio2weight_tip20(targets, self.class_ratios)
                loss_matrix = loss_matrix * weights
            elif self.weight_method == 'focal':
                # Focal loss style weighting
                probs = torch.sigmoid(logits)
                pt = targets * probs + (1 - targets) * (1 - probs)
                focal_weights = (1 - pt) ** 2
                loss_matrix = loss_matrix * focal_weights
        
        # Apply reduction
        if self.size_sum:
            loss = loss_matrix.sum(dim=1).mean()
        else:
            if self.reduction == 'mean':
                loss = loss_matrix.mean()
            elif self.reduction == 'sum':
                loss = loss_matrix.sum()
            elif self.reduction == 'none':
                loss = loss_matrix
            else:
                raise ValueError(f"Unknown reduction: {self.reduction}")
        
        return loss, loss_matrix


class ScaledBCELoss(nn.Module):
    """
    Scaled Binary Cross Entropy Loss.
    
    This loss applies different scaling factors to positive and negative logits
    before computing BCE, which can help with gradient flow and convergence.
    
    Args:
        pos_scale: Scaling factor for positive samples
        neg_scale: Scaling factor for negative samples
        class_ratios: Class ratios for sample weighting
        reduction: Reduction method
        label_smoothing: Label smoothing factor
        size_sum: If True, sum over classes then mean over batch
    
    Example:
        >>> loss_fn = ScaledBCELoss(pos_scale=30.0, neg_scale=30.0,
        ...                         class_ratios=class_ratios)
        >>> loss, loss_matrix = loss_fn(logits, targets)
    """
    
    def __init__(self,
                 pos_scale: float = 30.0,
                 neg_scale: float = 30.0,
                 class_ratios: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 reduction: str = 'mean',
                 label_smoothing: float = 0.0,
                 size_sum: bool = True):
        super().__init__()
        
        self.pos_scale = pos_scale
        self.neg_scale = neg_scale
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.size_sum = size_sum
        
        if class_ratios is not None:
            if isinstance(class_ratios, np.ndarray):
                class_ratios = torch.from_numpy(class_ratios).float()
            self.register_buffer('class_ratios', class_ratios)
        else:
            self.class_ratios = None
    
    def forward(self, 
                logits: torch.Tensor, 
                targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with logit scaling."""
        
        # Handle list input
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        
        # Apply scaling to logits
        scaled_logits = (logits * targets * self.pos_scale + 
                        logits * (1 - targets) * self.neg_scale)
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = ((1 - self.label_smoothing) * targets + 
                      self.label_smoothing * (1 - targets))
        
        # Compute BCE loss
        loss_matrix = F.binary_cross_entropy_with_logits(
            scaled_logits, targets, reduction='none'
        )
        
        # Apply sample weighting if available
        if self.class_ratios is not None:
            weights = ratio2weight_tip20(targets, self.class_ratios)
            loss_matrix = loss_matrix * weights
        
        # Apply reduction
        if self.size_sum:
            loss = loss_matrix.sum(dim=1).mean()
        else:
            if self.reduction == 'mean':
                loss = loss_matrix.mean()
            elif self.reduction == 'sum':
                loss = loss_matrix.sum()
            else:
                loss = loss_matrix
        
        return loss, loss_matrix


# ============================================================================
# Configuration System
# ============================================================================

@dataclass
class LossConfig:
    """Configuration class for multi-label loss functions."""
    
    # Loss type
    loss_type: str = 'adaptive_bce'
    
    # Class balancing
    use_class_weights: bool = True
    weight_method: str = 'tip20'  # 'tip20', 'focal', 'inverse_freq'
    
    # Loss parameters
    label_smoothing: float = 0.0
    size_sum: bool = True
    reduction: str = 'mean'
    
    # Scaling parameters (for ScaledBCE)
    pos_scale: float = 30.0
    neg_scale: float = 30.0
    
    # Additional parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'loss_type': self.loss_type,
            'use_class_weights': self.use_class_weights,
            'weight_method': self.weight_method,
            'label_smoothing': self.label_smoothing,
            'size_sum': self.size_sum,
            'reduction': self.reduction,
            'pos_scale': self.pos_scale,
            'neg_scale': self.neg_scale,
            **self.extra_params
        }


class MultiLabelLossFactory:
    """Factory class for creating multi-label loss functions."""
    
    @staticmethod
    def create_loss(loss_type: str, 
                   class_ratios: Optional[Union[np.ndarray, torch.Tensor]] = None,
                   **kwargs) -> nn.Module:
        """
        Create a multi-label loss function.
        
        Args:
            loss_type: Type of loss ('bce', 'weighted_bce', 'adaptive_bce', 'scaled_bce')
            class_ratios: Class ratios for weighting
            **kwargs: Additional arguments for the loss function
        
        Returns:
            Loss function instance
        
        Example:
            >>> loss_fn = MultiLabelLossFactory.create_loss(
            ...     'adaptive_bce', 
            ...     class_ratios=class_ratios,
            ...     label_smoothing=0.1
            ... )
        """
        loss_type = loss_type.lower()
        
        if loss_type in ['bce', 'bceloss']:
            return nn.BCEWithLogitsLoss(**kwargs)
        
        elif loss_type in ['weighted_bce', 'weighted_bceloss']:
            pos_weights = None
            if class_ratios is not None:
                if isinstance(class_ratios, np.ndarray):
                    pos_weights = torch.from_numpy(1.0 / (class_ratios + 1e-8)).float()
                else:
                    pos_weights = 1.0 / (class_ratios + 1e-8)
            return WeightedBCELoss(pos_weights=pos_weights, **kwargs)
        
        elif loss_type in ['adaptive_bce', 'adaptive_bceloss']:
            return AdaptiveWeightedBCELoss(class_ratios=class_ratios, **kwargs)
        
        elif loss_type in ['scaled_bce', 'scaled_bceloss']:
            return ScaledBCELoss(class_ratios=class_ratios, **kwargs)
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


class LossConfigManager:
    """Manager class for different loss configurations."""
    
    @staticmethod
    def get_pedestrian_config() -> LossConfig:
        """Configuration optimized for pedestrian attribute recognition."""
        return LossConfig(
            loss_type='adaptive_bce',
            use_class_weights=True,
            weight_method='tip20',
            label_smoothing=0.1,
            size_sum=True,
            reduction='mean'
        )
    
    @staticmethod
    def get_general_multilabel_config() -> LossConfig:
        """General configuration for multi-label classification."""
        return LossConfig(
            loss_type='weighted_bce',
            use_class_weights=True,
            weight_method='inverse_freq',
            label_smoothing=0.05,
            size_sum=True,
            reduction='mean'
        )
    
    @staticmethod
    def get_balanced_config() -> LossConfig:
        """Configuration for relatively balanced datasets."""
        return LossConfig(
            loss_type='bce',
            use_class_weights=False,
            label_smoothing=0.0,
            size_sum=False,
            reduction='mean'
        )
    
    @staticmethod
    def get_focal_config() -> LossConfig:
        """Configuration using focal loss style weighting."""
        return LossConfig(
            loss_type='adaptive_bce',
            use_class_weights=True,
            weight_method='focal',
            label_smoothing=0.0,
            size_sum=True,
            reduction='mean'
        )
    
    @staticmethod
    def get_scaled_config() -> LossConfig:
        """Configuration for scaled BCE loss."""
        return LossConfig(
            loss_type='scaled_bce',
            use_class_weights=True,
            weight_method='tip20',
            pos_scale=30.0,
            neg_scale=30.0,
            label_smoothing=0.1,
            size_sum=True,
            reduction='mean'
        )


def create_loss_from_config(config: LossConfig, 
                           class_ratios: Optional[Union[np.ndarray, torch.Tensor]] = None) -> torch.nn.Module:
    """
    Create a loss function from configuration.
    
    Args:
        config: Loss configuration
        class_ratios: Class ratios for weighting (if applicable)
    
    Returns:
        Configured loss function
    
    Example:
        >>> config = LossConfigManager.get_pedestrian_config()
        >>> loss_fn = create_loss_from_config(config, class_ratios)
    """
    # Prepare parameters
    params = {
        'reduction': config.reduction,
        'label_smoothing': config.label_smoothing,
        'size_sum': config.size_sum
    }
    
    # Add class ratios if using class weights
    if config.use_class_weights and class_ratios is not None:
        params['class_ratios'] = class_ratios
    
    # Add scaling parameters for scaled BCE
    if config.loss_type in ['scaled_bce', 'scaled_bceloss']:
        params.update({
            'pos_scale': config.pos_scale,
            'neg_scale': config.neg_scale
        })
    
    # Add weight method for adaptive BCE
    if config.loss_type in ['adaptive_bce', 'adaptive_bceloss']:
        params['weight_method'] = config.weight_method
    
    # Add extra parameters
    params.update(config.extra_params)
    
    # Create loss function
    return MultiLabelLossFactory.create_loss(config.loss_type, **params)


# Predefined configurations for common scenarios
PEDESTRIAN_ATTR_CONFIG = LossConfigManager.get_pedestrian_config()
GENERAL_MULTILABEL_CONFIG = LossConfigManager.get_general_multilabel_config()
BALANCED_DATASET_CONFIG = LossConfigManager.get_balanced_config()
FOCAL_LOSS_CONFIG = LossConfigManager.get_focal_config()
SCALED_BCE_CONFIG = LossConfigManager.get_scaled_config()

# Configuration presets
LOSS_PRESETS = {
    'pedestrian': PEDESTRIAN_ATTR_CONFIG,
    'general': GENERAL_MULTILABEL_CONFIG,
    'balanced': BALANCED_DATASET_CONFIG,
    'focal': FOCAL_LOSS_CONFIG,
    'scaled': SCALED_BCE_CONFIG
}


def get_preset_config(preset_name: str) -> LossConfig:
    """Get a preset configuration by name."""
    if preset_name not in LOSS_PRESETS:
        available = list(LOSS_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    return LOSS_PRESETS[preset_name]


# ============================================================================
# Usage Examples and Testing
# ============================================================================

def example_basic_usage():
    """Basic usage example."""
    print("=== Basic Usage Example ===")
    
    # Create synthetic data
    batch_size, num_classes = 32, 26
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    # Simulate class ratios (imbalanced)
    class_ratios = np.random.beta(0.3, 1.5, num_classes)
    
    # Create loss function
    loss_fn = AdaptiveWeightedBCELoss(
        class_ratios=class_ratios,
        label_smoothing=0.1,
        size_sum=True
    )
    
    # Compute loss
    loss, loss_matrix = loss_fn(logits, targets)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss matrix shape: {loss_matrix.shape}")
    print(f"Class ratios (first 10): {class_ratios[:10].round(3)}")
    print()


def example_training_integration():
    """Training loop integration example."""
    print("=== Training Integration Example ===")
    
    # Model setup
    class SimpleModel(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        
        def forward(self, x):
            return self.classifier(x)
    
    # Setup
    input_dim, num_classes = 2048, 26
    model = SimpleModel(input_dim, num_classes)
    
    # Create class ratios
    class_ratios = np.random.beta(0.4, 1.2, num_classes)
    
    # Loss and optimizer
    loss_fn = AdaptiveWeightedBCELoss(class_ratios=class_ratios, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    
    # Training simulation
    model.train()
    for epoch in range(3):
        # Simulate batch
        batch_features = torch.randn(32, input_dim)
        batch_labels = torch.bernoulli(torch.from_numpy(class_ratios).expand(32, -1))
        
        # Forward pass
        logits = model(batch_features)
        loss, _ = loss_fn(logits, batch_labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
    print()


def example_preset_configs():
    """Preset configurations example."""
    print("=== Preset Configurations Example ===")
    
    # Test data
    batch_size, num_classes = 16, 20
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    class_ratios = np.random.beta(0.3, 1.5, num_classes)
    
    # Test different presets
    presets = ['pedestrian', 'general', 'focal', 'scaled']
    
    for preset_name in presets:
        config = get_preset_config(preset_name)
        loss_fn = create_loss_from_config(config, class_ratios)
        
        if hasattr(loss_fn, 'forward') and len(loss_fn.forward.__code__.co_varnames) > 3:
            loss_val, _ = loss_fn(logits, targets)
        else:
            loss_val = loss_fn(logits, targets)
        
        print(f"{preset_name:12s}: {loss_val.item():.4f}")
    print()


def example_pedestrian_attributes():
    """Pedestrian attribute recognition example."""
    print("=== Pedestrian Attributes Example ===")
    
    # Realistic pedestrian attribute ratios
    attribute_names = [
        'Male', 'Female', 'Young', 'Adult', 'Old',
        'Hat', 'Glasses', 'Backpack', 'Handbag', 'Shoes_Leather',
        'Shirt_Long', 'Shirt_Short', 'Pants_Long', 'Pants_Short', 'Skirt',
        'Dress', 'Coat', 'Jacket', 'Suit', 'Casual',
        'Formal', 'Sports', 'Uniform', 'Traditional', 'Other', 'Accessory'
    ]
    
    class_ratios = np.array([
        0.52, 0.48, 0.25, 0.60, 0.15,  # Demographics
        0.15, 0.20, 0.30, 0.25, 0.40,  # Accessories
        0.35, 0.45, 0.70, 0.25, 0.15,  # Clothing types
        0.20, 0.30, 0.25, 0.10, 0.50,  # Clothing styles
        0.15, 0.20, 0.05, 0.03, 0.08, 0.12   # Special categories
    ])
    
    # Optimized loss for pedestrian attributes
    loss_fn = AdaptiveWeightedBCELoss(
        class_ratios=class_ratios,
        weight_method='tip20',
        label_smoothing=0.1,
        size_sum=True
    )
    
    # Test batch
    batch_size = 32
    logits = torch.randn(batch_size, len(attribute_names))
    targets = torch.bernoulli(torch.from_numpy(class_ratios).expand(batch_size, -1))
    
    loss, loss_matrix = loss_fn(logits, targets)
    
    print(f"Batch loss: {loss.item():.4f}")
    print(f"Rare attributes (ratio < 0.1): {np.sum(class_ratios < 0.1)} out of {len(class_ratios)}")
    print(f"Per-attribute loss range: [{loss_matrix.mean(0).min():.4f}, {loss_matrix.mean(0).max():.4f}]")
    print()


# ============================================================================
# Main Testing Function
# ============================================================================

if __name__ == "__main__":
    print("Complete Multi-Label Loss Functions")
    print("=" * 50)
    print()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run examples
    example_basic_usage()
    example_training_integration()
    example_preset_configs()
    example_pedestrian_attributes()
    
    print("=" * 50)
    print("Quick Start Guide:")
    print("1. For pedestrian attributes: use 'pedestrian' preset")
    print("2. For general multi-label: use 'general' preset")
    print("3. For balanced datasets: use 'balanced' preset")
    print("4. For custom needs: create LossConfig manually")
    print()
    print("Example:")
    print(">>> from complete_multilabel_losses import *")
    print(">>> config = get_preset_config('pedestrian')")
    print(">>> loss_fn = create_loss_from_config(config, class_ratios)")
    print(">>> loss, loss_matrix = loss_fn(logits, targets)")
    print()
    print("All tests completed successfully!")