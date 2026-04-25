"""
train_and_generate_v2.py - VAE training and synthetic data generation

Utilities for VAE training, synthetic sample generation, and downstream
quality evaluation.
"""

import os
import shutil
import math
from models import allocate_synthetic_budget
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
import numpy as np
import random
from typing import Dict, Tuple, Optional, Any

# Import the evaluator and VAE model
from models.metrics import evaluate_fidelity_diversity, audit_synthetic
from models.braintumor.vae_braintumor import VAE

# Opacus for Differential Privacy
try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False

# Import relevant parts from non-DP modules
from models.braintumor.vae_brain2 import VAE as VAE_DP  # This one uses GroupNorm


# ───────────────────────────────────────────────────────────────────────────────
# Helper: wrap (N,3,64,64) tensor into a Dataset that emits ImageNet-normalized
# 299×299 tensors, ready for Inception‑v3 / ResNet backbones.
# ───────────────────────────────────────────────────────────────────────────────
class TensorImageDataset(Dataset):
    def __init__(self, tensor: torch.Tensor, target_size: int = None):
        """
        tensor: shape (N,C,H,W) with values in [0,1] or [0,255].
        If target_size is None, keeps original size and only normalizes.
        """
        self.tensor = tensor
        num_channels = tensor.shape[1]

        if target_size is None:
            # Keep native resolution, only normalize
            # Backbones always expect 3 channels
            self.prep = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            self.resize = False
        else:
            # Resize and normalize (for compatibility)
            self.to_pil = transforms.ToPILImage()
            self.prep = transforms.Compose([
                transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
                # Grayscale to RGB conversion happens in __getitem__ if needed
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.resize = True

    def __len__(self):
        return self.tensor.size(0)

    def __getitem__(self, idx):
        img = self.tensor[idx]

        # If model expects 3 channels (Inception/ResNet) but data is 1 channel
        if self.prep is not None:
            # Check if it's a normalization transform or a full pipeline
            if hasattr(self.prep, "transforms"):
                # Full pipeline (includes resize/normalize)
                img_pil = self.to_pil(img)
                # ToPILImage will handle 1-channel or 3-channel
                # But transforms.ToTensor() will preserve 1-channel if PIL is grayscale
                # So we must convert to RGB if we want 3 channels for backbone
                if self.tensor.shape[1] == 1:
                    img_pil = img_pil.convert("RGB")
                img = self.prep(img_pil)
            else:
                # Just normalization
                if self.tensor.shape[1] == 1:
                    # Expand to 3 channels for backbone compatibility
                    img = img.expand(3, -1, -1)
                img = self.prep(img)
        return img, 0


# ───────────────────────────────────────────────────────────────────────────────
# Main function
# ───────────────────────────────────────────────────────────────────────────────
def train_and_generate_by_label(
        data_tensor, label_tensor,
        synthetic_ep: int = 500,
        synthetic_batch_size: int = 128,
        synthetic_latent_dim: int = 64,
        synthetic_count: int = 100,
        synthetic_output_dir: str = "synthetic_images",
        save_real: bool = False,
        real_output_dir: str = "real_images",
        img_size: int = 64,
        num_img_channels: int = 3,
        scale_syn: bool = False,
        do_audit: bool = True,
        target_size: int = None,
        seed: int = 42,
        **kwargs):
    """
    Train a label-conditional VAE and generate synthetic samples, then
    evaluate alpha-Precision, beta-Recall, Authenticity between the entire real
    dataset and the generated one.

    Args:
        data_tensor: Input images (N, C, H, W) with values in [0, 1]
        label_tensor: Labels (N,) with integer class labels
        synthetic_ep: Number of training epochs per label (default: 500)
        synthetic_batch_size: Batch size during VAE training (default: 128)
        synthetic_latent_dim: Latent dimension for VAE (default: 64)
        synthetic_count: Number of synthetic images to generate per label (default: 100)
        synthetic_output_dir: Output directory for synthetic images
        save_real: Whether to save real images (default: False)
        real_output_dir: Output directory for real images
        img_size: Image size (default: 64)
        num_img_channels: Number of image channels (default: 3)
        scale_syn: If True, scale synthetic count by label proportion (default: False)
        do_audit: Apply quality auditing/filtering (default: True)
        target_size: Target size for metrics (native if None)
        seed: Random seed for reproducibility (default: 42)
        **kwargs: Additional arguments (ignored for compatibility)

    Returns
    -------
    synthetic_x : torch.Tensor of shape (M, C, H, W)
    synthetic_y : torch.Tensor of shape (M,)
    metrics     : dict with {'alpha_precision', 'beta_recall', 'authenticity'}
    """
    # Set random seeds for reproducibility.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Clean previous outputs
    if os.path.exists(synthetic_output_dir):
        shutil.rmtree(synthetic_output_dir)
    if save_real and os.path.exists(real_output_dir):
        shutil.rmtree(real_output_dir)

    synthetic_images_list, synthetic_labels_list = [], []
    unique_labels = torch.unique(label_tensor)
    print(f"[INFO] Found labels: {unique_labels.tolist()}")

    label_budgets = allocate_synthetic_budget(label_tensor, data_tensor, synthetic_count, scale_syn)

    for lbl in unique_labels:
        # ---------------------------------------------------------------------
        # 1. subset + bookkeeping
        # ---------------------------------------------------------------------
        mask = (label_tensor == lbl)
        subset_data = data_tensor[mask]
        # Match original FedAug semantics: scale per label by its proportion on the client
        synth_num = label_budgets[int(lbl.item())]
        print(f"[INFO] Label {lbl.item()}: {subset_data.shape[0]} real -> {synth_num} synthetic")

        # optional: save real images
        if save_real:
            label_real_dir = os.path.join(real_output_dir, f"label_{lbl.item()}")
            os.makedirs(label_real_dir, exist_ok=True)
            for i, img in enumerate(subset_data.cpu()):
                plt.imsave(os.path.join(label_real_dir, f"real_{i}.png"),
                           img.permute(1, 2, 0).numpy())

        # ---------------------------------------------------------------------
        # 2. train VAE on this label
        # ---------------------------------------------------------------------
        dl = DataLoader(TensorDataset(subset_data), batch_size=synthetic_batch_size, shuffle=True)

        model = VAE(num_latent_dims=synthetic_latent_dim,
                    num_img_channels=num_img_channels,
                    max_num_filters=128,
                    device=device,
                    img_size=img_size).to(device)
        optimiser = optim.Adam(model.parameters(), lr=1e-3)

        model.train()

        for epoch in range(synthetic_ep):
            total_loss = 0.0
            for (x,) in dl:
                x = x.to(device)
                optimiser.zero_grad()
                # forward
                x_hat = model(x)

                # Compute reconstruction loss (using MSE loss here)
                recon_loss = F.mse_loss(x_hat, x, reduction="sum")

                # Original FedAug loss: reconstruction + full KL (no annealing)
                loss = recon_loss + model.kl_div
                loss.backward()
                optimiser.step()
                total_loss += loss.item()
            if (epoch + 1) % max(1, synthetic_ep // 10) == 0:
                print(f"[train] label {lbl.item()}, epoch {epoch + 1}/{synthetic_ep}, "
                      f"avg loss={total_loss / len(dl.dataset):.4f}")

        # ---------------------------------------------------------------------
        # 3. sample synthetic images
        # ---------------------------------------------------------------------
        model.eval()
        with torch.no_grad():
            z = torch.randn(synth_num, synthetic_latent_dim, device=device)
            synth_imgs = model.decode(z).cpu()

        synthetic_images_list.append(synth_imgs)
        synthetic_labels_list.append(torch.full((synth_imgs.size(0),), lbl.item(), dtype=torch.long))
        print(f"[gen ] label {lbl.item()}: generated {synth_imgs.size(0)} synthetic images")

    # -------------------------------------------------------------------------
    # 4. concatenate across labels
    # -------------------------------------------------------------------------
    synthetic_x = torch.cat(synthetic_images_list, dim=0).clamp(0.0, 1.0)
    synthetic_y = torch.cat(synthetic_labels_list, dim=0)
    print(f"[INFO] Total synthetic samples: {synthetic_x.shape[0]}")

    # Use native resolution - no upscaling by default
    real_ds = TensorImageDataset(data_tensor, target_size=target_size)
    fake_ds = TensorImageDataset(synthetic_x, target_size=target_size)

    metrics = evaluate_fidelity_diversity(
        real_ds, fake_ds,
        alpha=0.90, beta=0.90,
        batch_size=256,
        backbone="resnet18",
        input_size=img_size
    )

    print(
        "[METRICS] alpha-Precision={alpha_precision:.3f}, "
        "beta-Recall={beta_recall:.3f}, Authenticity={authenticity:.3f}"
        .format(**metrics)
    )

    if do_audit:
        fake_ds_audit = TensorImageDataset(synthetic_x, target_size=None)
        audit = audit_synthetic(
            real_ds,
            fake_ds_audit,
            alpha=0.90, beta=0.90,
            batch_size=256,
            backbone="resnet18",
            input_size=img_size,
            rule="precision_and_auth",
        )

        kept = int(audit["keep_mask"].sum())
        total = len(audit["keep_mask"])
        print(f"[AUDIT] kept {kept}/{total} synthetic images "
              f"({kept / total:.1%}) using precision∧authenticity rule")
        print("[AUDIT] metrics after audit:", audit["metrics_after"])

        # prune tensors for downstream use
        synthetic_x = synthetic_x[audit["keep_mask"]]
        synthetic_y = synthetic_y[audit["keep_mask"]]
        metrics_after = audit["metrics_after"]
    else:
        metrics_after = metrics

    return synthetic_x, synthetic_y, metrics_after


def train_dp_vae(
    data_tensor: torch.Tensor,
    epochs: int,
    batch_size: int,
    latent_dim: int,
    max_filters: int,
    lr: float,
    noise_multiplier: float,
    max_grad_norm: float,
    kl_warmup: int,
    device,
    img_size: int,
    num_img_channels: int,
    delta: float = 1e-5,
    **kwargs
) -> Tuple[nn.Module, list, float]:
    """
    Trains a VAE with DP-SGD using Opacus.
    """
    if not OPACUS_AVAILABLE:
        raise ImportError("Opacus is not installed. Please install it to use DP-FedAug.")

    dl = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=True)

    # Use the DP-compliant VAE (GroupNorm)
    model = VAE_DP(latent_dim, num_img_channels, max_filters, device, img_size)

    # Ensure model is DP-compliant
    errors = ModuleValidator.validate(model, strict=False)
    if len(errors) > 0:
        model = ModuleValidator.fix(model)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if noise_multiplier > 0:
        privacy_engine = PrivacyEngine()
        model, optimizer, dl = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dl,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )
    else:
        privacy_engine = None

    eps_history = []
    model.train()
    last_loss = 0.0

    for epoch in range(epochs):
        total_loss = 0.0
        for (x,) in dl:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat = model(x)

            # Reconstruction loss
            recon_loss = F.mse_loss(x_hat, x, reduction="sum")

            # KL Divergence
            kl_div = model.kl_div if hasattr(model, "kl_div") else model._module.kl_div
            kl_weight = min(1.0, epoch / max(1, kl_warmup))

            loss = recon_loss + kl_weight * kl_div
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if privacy_engine and noise_multiplier > 0:
            epsilon = privacy_engine.get_epsilon(delta)
        else:
            epsilon = float("inf")

        eps_history.append(epsilon)
        last_loss = total_loss / len(dl.dataset) if len(dl.dataset) > 0 else 0.0

    return model, eps_history, last_loss


def train_and_generate_by_label_dp(
    data_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    synthetic_ep: int = 500,
    synthetic_batch_size: int = 128,
    synthetic_latent_dim: int = 64,
    synthetic_count: int = 100,
    img_size: int = 64,
    num_img_channels: int = 3,
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0,
    kl_warmup: int = 100,
    delta: float = 1e-5,
    scale_syn: bool = False,
    do_audit: bool = False,
    seed: int = 42,
    target_size: Optional[int] = None,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float], Dict[int, float]]:
    """
    DP version of train_and_generate_by_label.
    Trains a separate DP-VAE per label.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    synthetic_images_list, synthetic_labels_list = [], []
    epsilon_per_label = {}

    unique_labels = torch.unique(label_tensor)
    print(f"[DP-FedAug] Found labels: {unique_labels.tolist()}")

    label_budgets = allocate_synthetic_budget(label_tensor, data_tensor, synthetic_count, scale_syn)

    for lbl in unique_labels:
        mask = (label_tensor == lbl)
        subset_data = data_tensor[mask]

        # Scaling logic same as non-DP
        synth_num = label_budgets[int(lbl.item())]

        print(f"[DP-FedAug] Label {lbl.item()}: {subset_data.shape[0]} real -> {synth_num} synthetic")

        # Train DP-VAE
        model, eps_history, last_loss = train_dp_vae(
            data_tensor=subset_data,
            epochs=synthetic_ep,
            batch_size=synthetic_batch_size,
            latent_dim=synthetic_latent_dim,
            max_filters=kwargs.get("max_filters", 128),
            lr=kwargs.get("lr", 1e-3),
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            kl_warmup=kl_warmup,
            device=device,
            img_size=img_size,
            num_img_channels=num_img_channels,
            delta=delta,
        )

        eps_last = eps_history[-1] if eps_history else float("nan")
        epsilon_per_label[int(lbl.item())] = eps_last
        print(f"[DP-FedAug] Label {lbl.item()} complete. Epsilon: {eps_last:.2f}, Final Loss: {last_loss:.4f}")

        # Sample synthetic images
        model.eval()
        decoder_func = model.decode if hasattr(model, "decode") else model._module.decode
        with torch.no_grad():
            z = torch.randn(synth_num, synthetic_latent_dim, device=device)
            synth_imgs = decoder_func(z).cpu().clamp(0.0, 1.0)

        synthetic_images_list.append(synth_imgs)
        synthetic_labels_list.append(torch.full((synth_imgs.size(0),), lbl.item(), dtype=torch.long))

    synthetic_x = torch.cat(synthetic_images_list, dim=0)
    synthetic_y = torch.cat(synthetic_labels_list, dim=0)

    # Evaluate quality metrics (using 3-channel conversion if needed, handled by TensorImageDataset)
    real_ds = TensorImageDataset(data_tensor, target_size=target_size)
    fake_ds = TensorImageDataset(synthetic_x, target_size=target_size)

    metrics = evaluate_fidelity_diversity(
        real_ds, fake_ds,
        alpha=0.90, beta=0.90,
        batch_size=256,
        backbone="resnet18",
        input_size=img_size,
        device=device,
    )

    return synthetic_x, synthetic_y, metrics, epsilon_per_label
