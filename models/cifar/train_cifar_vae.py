import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import math
from typing import Dict, Tuple, Optional
from contextlib import nullcontext
from models import allocate_synthetic_budget
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from models.cifar.vae_cifar_pp import CIFAR_VAE_PP, augment_batch_repeat
from models.metrics import evaluate_fidelity_diversity
from models.train_and_generate import TensorImageDataset


def _resolve_device(device: Optional[torch.device] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_cifar_vae_dp(
    data_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    epochs: int,
    batch_size: int,
    latent_dim: int,
    noise_multiplier: float,
    max_grad_norm: float,
    kl_warmup: int,
    lr: float,
    delta: float,
    img_size: int,
    scale_syn: bool,
    synthetic_count: int,
    seed: int,
    eval_metrics: bool = True,
    device: Optional[torch.device] = None,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float], Dict[int, float]]:
    """
    Trains a separate DP-VAE++ (DDPM-style) for each CIFAR-10 label and generates synthetic data.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = _resolve_device(device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    synthetic_images_list, synthetic_labels_list = [], []
    epsilon_per_label = {}

    unique_labels = torch.unique(label_tensor)
    print(f"[CIFAR10-DP-FedAug] Found labels: {unique_labels.tolist()}")

    # Training parameters following De et al. (2022) methodology
    augmult = 4
    max_physical_batch_size = 32
    ema_decay = 0.9999  # De et al. Appendix C.3

    label_budgets = allocate_synthetic_budget(label_tensor, data_tensor, synthetic_count, scale_syn)

    for lbl in unique_labels:
        mask = (label_tensor == lbl)
        subset_data = data_tensor[mask]

        synth_num = label_budgets[int(lbl.item())]
        print(f"[CIFAR10-DP-FedAug] Label {lbl.item()}: {subset_data.shape[0]} real -> {synth_num} synthetic")

        model = CIFAR_VAE_PP(latent_dim=latent_dim, img_channels=3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # We use drop_last=False so we don't lose data, but Opacus might complain if batch_size > dataset_size
        # The client_app.py already caps sample_rate at 0.99
        dl = DataLoader(TensorDataset(subset_data), batch_size=batch_size, shuffle=True, drop_last=False)

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

        # EMA setup (De et al. Sec 3.1, Appendix C.3)
        base_ref = model._module if (privacy_engine and noise_multiplier > 0) else model
        ema_state = {n: p.data.clone() for n, p in base_ref.named_parameters()}
        step_count = 0

        model.train()
        for epoch in range(epochs):
            # KL warmup schedule from DDPM-style notebook
            kl_weight = min(1.0, (epoch + 1) / max(10, epochs // 4))

            if privacy_engine and noise_multiplier > 0:
                ctx = BatchMemoryManager(
                    data_loader=dl, max_physical_batch_size=max_physical_batch_size, optimizer=optimizer
                )
            else:
                ctx = nullcontext(dl)

            with ctx as active_dl:
                for (xb,) in active_dl:
                    xb = xb.to(device)
                    b = xb.size(0)

                    # Apply augmentation multiplicity (De et al. Eq. 4)
                    x_aug = augment_batch_repeat(xb, augmult=augmult, pad=4)

                    recon, mu, logvar = model(x_aug)

                    # Compute loss per sample across the augmented repeats
                    recon_per = F.mse_loss(recon, x_aug, reduction='none').view(b * augmult, -1).sum(dim=1)
                    kl_per = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)

                    # Average over the augmult dimension (De et al. Eq. 4)
                    recon_per = recon_per.view(b, augmult).mean(dim=1)
                    kl_per = kl_per.view(b, augmult).mean(dim=1)

                    loss = (recon_per + kl_weight * kl_per).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # EMA update with warmup (De et al. Appendix C.3)
                    step_count += 1
                    decay_t = min(ema_decay, (1 + step_count) / (10 + step_count))
                    for n, p in base_ref.named_parameters():
                        if n in ema_state:
                            ema_state[n].mul_(decay_t).add_(p.data, alpha=1.0 - decay_t)

        # Copy EMA weights to model (De et al. Sec 3.1)
        for n, p in base_ref.named_parameters():
            if n in ema_state:
                p.data.copy_(ema_state[n])

        if privacy_engine and noise_multiplier > 0:
            epsilon = privacy_engine.get_epsilon(delta)
        else:
            epsilon = float("inf")
        epsilon_per_label[int(lbl.item())] = epsilon
        eps_str = f"{epsilon:.2f}" if epsilon < float("inf") else "∞"
        print(f"[CIFAR10-DP-FedAug] Label {lbl.item()} complete. ε={eps_str}")

        model.eval()
        decoder_func = model.decode if hasattr(model, "decode") else model._module.decode
        with torch.no_grad():
            z = torch.randn(synth_num, latent_dim, device=device)
            # The VAE++ decoder already has a Sigmoid, so output is in [0, 1]
            synth_imgs = decoder_func(z).cpu().clamp(0.0, 1.0)

        synthetic_images_list.append(synth_imgs)
        synthetic_labels_list.append(torch.full((synth_imgs.size(0),), lbl.item(), dtype=torch.long))

    synthetic_x = torch.cat(synthetic_images_list, dim=0)
    synthetic_y = torch.cat(synthetic_labels_list, dim=0)

    if eval_metrics:
        real_ds = TensorImageDataset(data_tensor, target_size=None)
        fake_ds = TensorImageDataset(synthetic_x, target_size=None)

        metrics = evaluate_fidelity_diversity(
            real_ds, fake_ds,
            alpha=0.90, beta=0.90,
            batch_size=256,
            backbone="resnet18",
            input_size=img_size,
            device=device
        )
    else:
        metrics = {
            "alpha_precision": float("nan"),
            "beta_recall": float("nan"),
            "authenticity": float("nan"),
        }

    return synthetic_x, synthetic_y, metrics, epsilon_per_label
