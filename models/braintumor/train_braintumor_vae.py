import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import math
from typing import Dict, Tuple
from opacus import PrivacyEngine
from models import allocate_synthetic_budget
from models.braintumor.vae_brain2 import VAE as BrainTumor_VAE
from models.metrics import evaluate_fidelity_diversity
from models.train_and_generate import TensorImageDataset

def train_braintumor_vae_dp(
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
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float], Dict[int, float]]:
    """
    Trains a separate DP-VAE for each Brain Tumor label and generates synthetic data.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    synthetic_images_list, synthetic_labels_list = [], []
    epsilon_per_label = {}
    
    unique_labels = torch.unique(label_tensor)
    print(f"[BrainTumor-DP-FedAug] Found labels: {unique_labels.tolist()}")

    label_budgets = allocate_synthetic_budget(label_tensor, data_tensor, synthetic_count, scale_syn)

    for lbl in unique_labels:
        mask = (label_tensor == lbl)
        subset_data = data_tensor[mask]

        synth_num = label_budgets[int(lbl.item())]
        
        print(f"[BrainTumor-DP-FedAug] Label {lbl.item()}: {subset_data.shape[0]} real -> {synth_num} synthetic")

        # 1. Create Model
        model = BrainTumor_VAE(num_latent_dims=latent_dim, num_img_channels=3, max_num_filters=128, device=device, img_size=img_size)
        model = model.to(device)
        
        # 2. Setup Optimizer and Data
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        dl = DataLoader(TensorDataset(subset_data), batch_size=batch_size, shuffle=True)

        # 3. Apply DP if noise_multiplier > 0
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

        # 4. Training Loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for (x,) in dl:
                x = x.to(device)
                optimizer.zero_grad()
                x_hat = model(x)
                
                recon_loss = F.mse_loss(x_hat, x, reduction='sum')
                kl_div = model.kl_div if hasattr(model, 'kl_div') else model._module.kl_div
                kl_weight = min(1.0, epoch / max(1, kl_warmup))
                
                loss = recon_loss + kl_weight * kl_div
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # 5. Privacy Accounting
        if privacy_engine and noise_multiplier > 0:
            epsilon = privacy_engine.get_epsilon(delta)
        else:
            epsilon = float('inf')
        epsilon_per_label[int(lbl.item())] = epsilon
        print(f"[BrainTumor-DP-FedAug] Label {lbl.item()} complete. Epsilon: {epsilon:.2f}")

        # 6. Generate Samples
        model.eval()
        decoder_func = model.decode if hasattr(model, 'decode') else model._module.decode
        with torch.no_grad():
            z = torch.randn(synth_num, latent_dim, device=device)
            synth_imgs = decoder_func(z).cpu().clamp(0.0, 1.0)

        synthetic_images_list.append(synth_imgs)
        synthetic_labels_list.append(torch.full((synth_imgs.size(0),), lbl.item(), dtype=torch.long))

    synthetic_x = torch.cat(synthetic_images_list, dim=0)
    synthetic_y = torch.cat(synthetic_labels_list, dim=0)

    # 7. Quality Metrics
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

    return synthetic_x, synthetic_y, metrics, epsilon_per_label
