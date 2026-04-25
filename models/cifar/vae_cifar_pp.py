"""
CIFAR VAE++ model and augmentation utilities for DP-SGD training.

Extracted from experiments/cifar/run_cifar_saliency_dpddpm_style.py
so that both the saliency script and the ablation script can share
the same model definition.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR_VAE_PP(nn.Module):
    """Wider CIFAR VAE with GroupNorm for DP compatibility."""

    def __init__(self, latent_dim: int = 96, img_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, stride=2, padding=1),
            nn.GroupNorm(16, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.SiLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GroupNorm(16, 64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, img_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z).view(-1, 256, 4, 4)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def augment_batch_repeat(x: torch.Tensor, augmult: int = 4, pad: int = 4) -> torch.Tensor:
    """Repeat each sample augmult times and apply crop/flip augmentations."""
    b, c, h, w = x.shape
    x_rep = x.unsqueeze(1).repeat(1, augmult, 1, 1, 1).reshape(b * augmult, c, h, w)

    if pad > 0:
        x_pad = F.pad(x_rep, (pad, pad, pad, pad), mode="reflect")
        max_off = 2 * pad
        y0 = torch.randint(0, max_off + 1, (x_rep.size(0),), device=x_rep.device)
        x0 = torch.randint(0, max_off + 1, (x_rep.size(0),), device=x_rep.device)
        crops = []
        for i in range(x_rep.size(0)):
            yi, xi = int(y0[i].item()), int(x0[i].item())
            crops.append(x_pad[i : i + 1, :, yi : yi + h, xi : xi + w])
        x_rep = torch.cat(crops, dim=0)

    flip_mask = torch.rand(x_rep.size(0), device=x_rep.device) < 0.5
    if flip_mask.any():
        x_rep[flip_mask] = torch.flip(x_rep[flip_mask], dims=[3])
    return x_rep
