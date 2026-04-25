"""
Shared Neural Network Models for Federated Learning

This module contains the model architectures used across different FL strategies.
All models are implemented according to the thesis specifications (Chapter 4).

Classification Models (Section 4.4):
- BrainTumorCNN: CNN for brain tumor detection

Generative Models (Section 4.6 - FedAug):
- BrainTumorVAE: VAE for synthetic brain MRI generation
- train_vae: Training function for BrainTumorVAE
- elbo_loss: ELBO loss computation for BrainTumorVAE
"""

import torch


def allocate_synthetic_budget(label_tensor: torch.Tensor, data_tensor: torch.Tensor,
                              synthetic_count: int, scale_syn: bool) -> dict:
    """Allocate synthetic sample budget across labels using floor + largest-remainder.

    When scale_syn is True, allocates proportionally to each label's share of the
    data, using the largest-remainder method to ensure the total sums exactly to
    synthetic_count. When False, each label gets synthetic_count samples.

    Returns a dict mapping label (int) -> number of synthetic samples.
    """
    unique_labels = torch.unique(label_tensor)
    if not scale_syn:
        return {int(lbl.item()): synthetic_count for lbl in unique_labels}

    n_total = data_tensor.shape[0]
    # Compute ideal (real-valued) allocations
    ideal = {}
    for lbl in unique_labels:
        n_lbl = int((label_tensor == lbl).sum().item())
        ideal[int(lbl.item())] = n_lbl / n_total * synthetic_count

    # Floor each allocation
    floored = {lbl: int(v) for lbl, v in ideal.items()}
    remainder = synthetic_count - sum(floored.values())

    # Distribute remainder to labels with the largest fractional parts
    frac_parts = sorted(ideal.keys(), key=lambda lbl: ideal[lbl] - floored[lbl], reverse=True)
    for i in range(remainder):
        floored[frac_parts[i]] += 1

    return floored

from models.cifar.cifar_cnn import CIFAR_CNN
from models.mnist.mnist_cnn import MNIST_CNN
from models.mnist.vae_mnist import MNIST_VAE
from models.braintumor.brain_tumor_cnn import BrainTumorCNN
from models.braintumor.vae_brain2 import VAE as BrainTumorVAE_DP
# Note: vae_braintumor.py exports VAE class (not BrainTumorVAE)
# Commenting out for now since it's not used in baseline FedAvg
# from .vae_braintumor import VAE, train_vae, elbo_loss
# from .timevae import TimeVAE, train_timevae, timevae_loss  # Commented out - only working on Brain Tumor dataset

__all__ = [
    # Utilities
    "allocate_synthetic_budget",
    # Classification models
    "BrainTumorCNN",
    "MNIST_CNN",
    "CIFAR_CNN",
    # Generative models - not used in baseline FedAvg
    # "VAE",
    # "TimeVAE",  # Commented out - only working on Brain Tumor dataset
    # Training functions - not used in baseline FedAvg
    # "train_vae",
    # "train_timevae",  # Commented out - only working on Brain Tumor dataset
    # Loss functions - not used in baseline FedAvg
    # "elbo_loss",
    # "timevae_loss"  # Commented out - only working on Brain Tumor dataset
]
