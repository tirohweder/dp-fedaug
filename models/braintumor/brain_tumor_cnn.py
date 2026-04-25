"""
Brain Tumor CNN Model - Convolutional Neural Network for Binary Classification

This module implements the CNN architecture for brain tumor detection as described
in Chapter 4, Section 4.4 of the thesis: "Brain Tumor - Convolutional Neural Network Classifier"

The architecture is inspired by the work of [4], with adjustments for improved performance:
- Group Normalization instead of Batch Normalization (better for small batches [27])
- Dropout regularization (p=0.3) to prevent overfitting

Architecture Details:
--------------------
Given an input MRI slice x ∈ R^(C×H×W), the network computes successive feature maps
through three convolutional blocks:

Block 1: h^(1) = MaxPool(ReLU(GN_G(W^(1) ⊗ x)))
Block 2: h^(2) = MaxPool(ReLU(GN_G(W^(2) ⊗ h^(1))))
Block 3: h^(3) = ReLU(GN_G(W^(3) ⊗ h^(2)))

where ⊗ denotes 2D convolution and GN_G(·) is group normalization with G groups.

Spatial dimensions are halved after blocks 1 and 2, while block 3 preserves resolution.
A channel-wise Global Average Pooling (GAP) then maps h^(3) to a feature vector:

z = GAP(h^(3))

which is regularized by dropout:

z̃ = Dropout_{p=0.3}(z)

Finally, a fully connected (linear) classification layer produces a logit s ∈ R,
converted to a probability via the sigmoid:

s = w^T z̃ + b,  ŷ = σ(s) = 1/(1 + e^(-s))

This compact architecture enables end-to-end training by standard backpropagation,
yielding a tumor-presence probability ŷ for each input.

References:
-----------
[4] Original CNN architecture inspiration
[26] Code implementation reference
[27] Group Normalization paper (Wu & He, 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BrainTumorCNN(nn.Module):
    """
    CNN Model for Brain Tumor Image Classification (Binary Classification).

    This model implements the architecture described in Section 4.4 of the thesis.
    It takes RGB MRI slices (3×64×64) as input and outputs a single logit for
    binary tumor detection.

    Architecture:
        - Input: RGB MRI slice (3 × 64 × 64)
        - Conv Block 1: 3→32 channels, MaxPool (32 × 32 × 32)
        - Conv Block 2: 32→64 channels, MaxPool (64 × 16 × 16)
        - Conv Block 3: 64→128 channels, no pooling (128 × 16 × 16)
        - Global Average Pooling: (128 × 16 × 16) → (128)
        - Dropout: p=0.3
        - Linear: 128 → 1 (binary classification logit)

    Training:
        - Loss: BCEWithLogitsLoss (combines sigmoid and binary cross-entropy)
        - Optimizer: Adam with weight_decay=1e-4
        - Gradient clipping: max_norm=1.0

    Input:
        x: Tensor of shape (batch_size, 3, 64, 64)
           RGB MRI slices normalized to [0, 1]

    Output:
        logits: Tensor of shape (batch_size, 1)
                Raw logits (use sigmoid for probability)

    Example:
        >>> model = BrainTumorCNN()
        >>> x = torch.randn(32, 3, 64, 64)  # batch of 32 MRI slices
        >>> logits = model(x)
        >>> probs = torch.sigmoid(logits)  # convert to probabilities
        >>> preds = (probs > 0.5).float()  # threshold at 0.5
    """

    def __init__(self):
        super(BrainTumorCNN, self).__init__()

        # Convolutional Block 1: 3 → 32 channels
        # Input: 3×64×64, Output after conv: 32×64×64, After pool: 32×32×32
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1,  # Preserve spatial dimensions
            bias=False  # GroupNorm has learnable affine parameters
        )
        self.gn1 = nn.GroupNorm(
            num_groups=8,     # 8 groups for 32 channels (4 channels per group)
            num_channels=32
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves spatial dimensions

        # Convolutional Block 2: 32 → 64 channels
        # Input: 32×32×32, Output after conv: 64×32×32, After pool: 64×16×16
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.gn2 = nn.GroupNorm(
            num_groups=8,     # 8 groups for 64 channels (8 channels per group)
            num_channels=64
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves spatial dimensions

        # Convolutional Block 3: 64 → 128 channels
        # Input: 64×16×16, Output: 128×16×16 (preserves spatial resolution)
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.gn3 = nn.GroupNorm(
            num_groups=8,     # 8 groups for 128 channels (16 channels per group)
            num_channels=128
        )

        # Global Average Pooling (GAP)
        # Maps spatial features to a single value per channel
        # Input: 128×H×W → Output: 128×1×1
        self.pool3 = nn.AdaptiveAvgPool2d(1)

        # Dropout Regularization
        # Applied after GAP to prevent overfitting
        self.dropout = nn.Dropout(p=0.3)

        # Classification Head (Binary Classification)
        # Maps 128-dimensional feature vector to a single logit
        self.classification_head = nn.Linear(
            in_features=128,
            out_features=1
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 64, 64)
               RGB MRI slices normalized to [0, 1]

        Returns:
            logits: Tensor of shape (batch_size, 1)
                    Raw logits for binary classification
                    Use torch.sigmoid(logits) to get probabilities
        """
        # Block 1: h^(1) = MaxPool(ReLU(GN(W^(1) ⊗ x)))
        # 3×64×64 → 32×64×64 → 32×32×32
        x = self.conv1(x)           # Convolution
        x = self.gn1(x)             # Group Normalization
        x = F.relu(x)               # Activation
        x = self.pool1(x)           # Spatial downsampling

        # Block 2: h^(2) = MaxPool(ReLU(GN(W^(2) ⊗ h^(1))))
        # 32×32×32 → 64×32×32 → 64×16×16
        x = self.conv2(x)           # Convolution
        x = self.gn2(x)             # Group Normalization
        x = F.relu(x)               # Activation
        x = self.pool2(x)           # Spatial downsampling

        # Block 3: h^(3) = ReLU(GN(W^(3) ⊗ h^(2)))
        # 64×16×16 → 128×16×16 (preserves spatial resolution)
        x = self.conv3(x)           # Convolution
        x = self.gn3(x)             # Group Normalization
        x = F.relu(x)               # Activation

        # Global Average Pooling: z = GAP(h^(3))
        # 128×16×16 → 128×1×1 → 128
        x = self.pool3(x)           # Global average pooling
        x = x.view(-1, 128)         # Flatten to (batch_size, 128)

        # Dropout Regularization: z̃ = Dropout(z)
        x = self.dropout(x)

        # Classification Head: s = w^T z̃ + b
        # 128 → 1
        logits = self.classification_head(x)

        return logits

    def get_num_parameters(self):
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_architecture_summary(self):
        """Returns a string summary of the architecture."""
        return f"""
BrainTumorCNN Architecture Summary:
-----------------------------------
Input: 3×64×64 RGB MRI slices

Block 1:
  Conv2d(3→32, kernel=3, padding=1)
  GroupNorm(8 groups, 32 channels)
  ReLU
  MaxPool2d(2×2) → 32×32×32

Block 2:
  Conv2d(32→64, kernel=3, padding=1)
  GroupNorm(8 groups, 64 channels)
  ReLU
  MaxPool2d(2×2) → 64×16×16

Block 3:
  Conv2d(64→128, kernel=3, padding=1)
  GroupNorm(8 groups, 128 channels)
  ReLU → 128×16×16

Global Average Pooling:
  AdaptiveAvgPool2d(1) → 128

Regularization:
  Dropout(p=0.3)

Classification:
  Linear(128→1)

Total Parameters: {self.get_num_parameters():,}
"""