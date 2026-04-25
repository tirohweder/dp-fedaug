#!/usr/bin/env python3
"""
Axes:
  target_eps, logical_batch_size, physical_batch_size, max_grad_norm,
  augmult, vae_epochs, latent_dim, vae_lr

Usage:
  # Dry-run to see grid
  python experiments/cifar/run_cifar_ablation_dpddpm.py --dry-run

  # Run specific axes locally
  python experiments/cifar/run_cifar_ablation_dpddpm.py --axes epsilon,augmult

  # Run all axes locally
  python experiments/cifar/run_cifar_ablation_dpddpm.py --execution-mode local

  # Generate HPC submission scripts
  python experiments/cifar/run_cifar_ablation_dpddpm.py --execution-mode hpc
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.cifar.cifar_cnn import CIFAR_CNN
from models.cifar.vae_cifar_pp import CIFAR_VAE_PP, augment_batch_repeat
from models.metrics import evaluate_fidelity_diversity
from models.metrics import TensorDataset as MetricsTensorDataset

# ---------------------------------------------------------------------------
# Baseline configuration
# ---------------------------------------------------------------------------
BASELINE = {
    "target_eps": 1.0,
    "logical_batch_size": 512,
    "physical_batch_size": 32,
    "max_grad_norm": 1.0,
    "augmult": 4,
    "vae_epochs": 40,
    "latent_dim": 96,
    "vae_lr": 1e-3,
}

SEEDS = [42, 43, 44]

# Fixed parameters (not ablated)
FIXED = {
    "n_real_per_class": 2000,
    "synthetic_count_per_class": 200,
    "delta": 1e-5,
    "classifier_epochs": 15,
    "classifier_lr": 1e-3,
    "classifier_batch_size": 64,
    "fewshot_per_class": [10, 50, 100],
    "test_batch_size": 256,
}

# Ablation axes: axis_name -> list of values
ABLATION_AXES: Dict[str, List[Any]] = {
    "target_eps": [0.5, 1.0, 2.0, 4.0, 8.0, None],  # None = no DP
    "logical_batch_size": [64, 128, 256, 512],
    "physical_batch_size": [8, 16, 32, 64],
    "max_grad_norm": [0.1, 0.5, 1.0, 2.0, 5.0],
    "augmult": [1, 2, 4],
    "vae_epochs": [10, 20, 40, 80],
    "latent_dim": [32, 64, 96, 128],
    "vae_lr": [5e-4, 1e-3, 2e-3, 5e-3],
}

# Map ablation axis name to the corresponding logged parameter key.
AXIS_TO_PARAM: Dict[str, str] = {
    "target_eps": "target_eps",
    "logical_batch_size": "logical_batch_size",
    "physical_batch_size": "physical_batch_size",
    "max_grad_norm": "max_grad_norm",
    "augmult": "augmult",
    "vae_epochs": "vae_epochs",
    "latent_dim": "latent_dim",
    "vae_lr": "vae_lr",
}

# HPC defaults (override via .env; see .env.example)
HPC_DEFAULTS = {
    "hpc_host": os.environ.get("HPC_HOST", "user@hpc.example.com"),
    "hpc_project_dir": os.environ.get("HPC_PROJECT_DIR", "~/dp-fedaug"),
    "hpc_queue": "gpua100",
    "hpc_walltime": "24:00",
    "hpc_gpus": 1,
    # DTU GPU queues require at least 4 CPU cores per GPU.
    "hpc_cores": 4,
    # Per-slot memory; with 4 cores this is 100GB total host RAM/job.
    "hpc_mem": "25GB",
}


# ---------------------------------------------------------------------------
# Config dataclass for a single run
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    # Ablation identity
    axis: str
    axis_value: Any
    seed: int
    # DP VAE params
    target_eps: Optional[float] = 1.0
    logical_batch_size: int = 512
    physical_batch_size: int = 64
    max_grad_norm: float = 1.0
    augmult: int = 4
    vae_epochs: int = 40
    latent_dim: int = 96
    vae_lr: float = 1e-3
    # Fixed params
    n_real_per_class: int = 2000
    synthetic_count_per_class: int = 200
    delta: float = 1e-5
    classifier_epochs: int = 15
    classifier_lr: float = 1e-3
    classifier_batch_size: int = 64
    fewshot_per_class: List[int] = field(default_factory=lambda: [10, 50, 100])
    test_batch_size: int = 256
    # Paths
    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "data")
    results_dir: Path = field(default_factory=lambda: REPO_ROOT / "results" / "cifar_ablation")

    @property
    def run_name(self) -> str:
        v = self.axis_value if self.axis_value is not None else "inf"
        return f"ablation_{self.axis}_{v}_seed{self.seed}"

    @property
    def use_dp(self) -> bool:
        return self.target_eps is not None


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------
def build_grid(axes: Optional[List[str]] = None) -> List[RunConfig]:
    """Build list of RunConfigs for all requested ablation axes."""
    if axes is None:
        axes = list(ABLATION_AXES.keys())

    configs: List[RunConfig] = []
    for axis in axes:
        if axis not in ABLATION_AXES:
            raise ValueError(f"Unknown axis '{axis}'. Choose from: {list(ABLATION_AXES.keys())}")
        for value in ABLATION_AXES[axis]:
            for seed in SEEDS:
                params = dict(BASELINE)
                params.update(FIXED)
                params[axis] = value
                # physical_batch_size must not exceed logical_batch_size
                if params["physical_batch_size"] > params["logical_batch_size"]:
                    params["physical_batch_size"] = params["logical_batch_size"]
                # Cap effective batch (physical × augmult) to avoid OOM
                # with Opacus per-sample gradients on long combined ablation runs.
                max_effective = 64
                augmult = params["augmult"]
                max_phys = max(max_effective // augmult, 8)
                if params["physical_batch_size"] > max_phys:
                    params["physical_batch_size"] = max_phys
                configs.append(RunConfig(
                    axis=axis,
                    axis_value=value,
                    seed=seed,
                    target_eps=params.get("target_eps"),
                    logical_batch_size=params["logical_batch_size"],
                    physical_batch_size=params["physical_batch_size"],
                    max_grad_norm=params["max_grad_norm"],
                    augmult=params["augmult"],
                    vae_epochs=params["vae_epochs"],
                    latent_dim=params["latent_dim"],
                    vae_lr=params["vae_lr"],
                    n_real_per_class=params["n_real_per_class"],
                    synthetic_count_per_class=params["synthetic_count_per_class"],
                    delta=params["delta"],
                    classifier_epochs=params["classifier_epochs"],
                    classifier_lr=params["classifier_lr"],
                    classifier_batch_size=params["classifier_batch_size"],
                    fewshot_per_class=params["fewshot_per_class"],
                    test_batch_size=params["test_batch_size"],
                ))
    return configs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cifar_data(
    cfg: RunConfig,
) -> Tuple[Dict[int, torch.Tensor], datasets.CIFAR10, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()])
    cifar_train = datasets.CIFAR10(root=cfg.data_dir, train=True, download=True, transform=transform)
    cifar_test = datasets.CIFAR10(root=cfg.data_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(cifar_test, batch_size=cfg.test_batch_size, shuffle=False)

    targets = np.array(cifar_train.targets)
    rng = np.random.default_rng(cfg.seed)

    real_by_label: Dict[int, torch.Tensor] = {}
    for lbl in range(10):
        indices = np.where(targets == lbl)[0]
        chosen = rng.choice(indices, size=cfg.n_real_per_class, replace=False)
        images = torch.stack([cifar_train[int(i)][0] for i in chosen], dim=0)
        real_by_label[lbl] = images

    return real_by_label, cifar_train, test_loader


# ---------------------------------------------------------------------------
# VAE training (DP and non-DP)
# ---------------------------------------------------------------------------
def train_vae_for_label(
    subset: torch.Tensor,
    lbl: int,
    cfg: RunConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Train a per-label VAE and return generated samples + training metrics."""
    n = subset.shape[0]
    ds = TensorDataset(subset)
    dl = DataLoader(ds, batch_size=cfg.logical_batch_size, shuffle=True, drop_last=False)

    model = CIFAR_VAE_PP(latent_dim=cfg.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.vae_lr)

    noise_multiplier = 0.0
    sample_rate = 0.0
    actual_eps = float("inf")

    if cfg.use_dp:
        sample_rate = min(cfg.logical_batch_size / max(n, 1), 0.99)
        noise_multiplier = float(get_noise_multiplier(
            target_epsilon=float(cfg.target_eps),
            target_delta=cfg.delta,
            sample_rate=sample_rate,
            epochs=cfg.vae_epochs,
        ))
        pe = PrivacyEngine()
        model, optimizer, dl = pe.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dl,
            noise_multiplier=noise_multiplier,
            max_grad_norm=cfg.max_grad_norm,
        )

    model.train()
    last_loss = float("nan")
    for epoch in range(cfg.vae_epochs):
        kl_weight = min(1.0, (epoch + 1) / max(10, cfg.vae_epochs // 4))
        epoch_loss = 0.0
        steps = 0

        if cfg.use_dp:
            ctx = BatchMemoryManager(
                data_loader=dl,
                max_physical_batch_size=cfg.physical_batch_size,
                optimizer=optimizer,
            )
        else:
            from contextlib import nullcontext
            ctx = nullcontext(dl)

        with ctx as active_dl:
            for (xb,) in active_dl:
                xb = xb.to(device)
                b = xb.size(0)
                x_aug = augment_batch_repeat(xb, augmult=cfg.augmult, pad=4)
                recon, mu, logvar = model(x_aug)

                recon_per = F.mse_loss(recon, x_aug, reduction="none").view(b * cfg.augmult, -1).sum(dim=1)
                kl_per = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
                recon_per = recon_per.view(b, cfg.augmult).mean(dim=1)
                kl_per = kl_per.view(b, cfg.augmult).mean(dim=1)
                loss = (recon_per + kl_weight * kl_per).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                steps += 1

        last_loss = epoch_loss / max(steps, 1)

    if cfg.use_dp:
        actual_eps = float(pe.get_epsilon(cfg.delta))

    # Generate synthetic samples
    model.eval()
    decoder_fn = model.decode if hasattr(model, "decode") else model._module.decode
    with torch.no_grad():
        z = torch.randn(cfg.synthetic_count_per_class, cfg.latent_dim, device=device)
        syn = decoder_fn(z).cpu().clamp(0, 1)

    metrics = {
        "label": lbl,
        "actual_epsilon": actual_eps,
        "noise_multiplier": noise_multiplier,
        "sample_rate": sample_rate,
        "final_loss": last_loss,
    }
    return syn, metrics


def train_all_labels(
    real_by_label: Dict[int, torch.Tensor],
    cfg: RunConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, float]]]:
    """Train per-label VAEs and collect all synthetic data."""
    syn_images: List[torch.Tensor] = []
    syn_labels: List[torch.Tensor] = []
    label_metrics: List[Dict[str, float]] = []

    for lbl in range(10):
        syn, metrics = train_vae_for_label(
            real_by_label[lbl], lbl, cfg, device,
        )
        syn_images.append(syn)
        syn_labels.append(torch.full((cfg.synthetic_count_per_class,), lbl, dtype=torch.long))
        label_metrics.append(metrics)
        print(f"  label={lbl} eps={metrics['actual_epsilon']:.3f} loss={metrics['final_loss']:.4f}")

    return torch.cat(syn_images), torch.cat(syn_labels), label_metrics


# ---------------------------------------------------------------------------
# Classifier evaluation
# ---------------------------------------------------------------------------
def train_and_eval_classifier(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
) -> float:
    model = CIFAR_CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(train_x, train_y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += int((pred == yb).sum().item())
            total += int(yb.size(0))
    return correct / max(total, 1)


def sample_fewshot(
    cifar_train: datasets.CIFAR10,
    n_per_class: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    targets = np.array(cifar_train.targets)
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for lbl in range(10):
        indices = np.where(targets == lbl)[0]
        chosen = rng.choice(indices, size=n_per_class, replace=False)
        xs.append(torch.stack([cifar_train[int(i)][0] for i in chosen]))
        ys.append(torch.full((n_per_class,), lbl, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


# ---------------------------------------------------------------------------
# Synthetic quality evaluation
# ---------------------------------------------------------------------------
def evaluate_quality(
    real_by_label: Dict[int, torch.Tensor],
    syn_x: torch.Tensor,
    syn_y: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Compute alpha-precision, beta-recall, authenticity using InceptionV3."""
    # Stack all real images
    real_all = torch.cat([real_by_label[lbl] for lbl in range(10)])
    real_labels = torch.cat([torch.full((real_by_label[lbl].shape[0],), lbl, dtype=torch.long) for lbl in range(10)])

    # Wrap as metric datasets (expects (image, label) tuples)
    real_ds = MetricsTensorDataset(real_all)
    fake_ds = MetricsTensorDataset(syn_x)

    metrics = evaluate_fidelity_diversity(
        real_source=real_ds,
        fake_source=fake_ds,
        backbone="resnet18",  # faster than inception for ablation
        device=device,
    )
    return metrics


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_single(
    cfg: RunConfig,
    device: torch.device,
    wb_run: Optional[wandb.sdk.wandb_run.Run],
) -> Dict[str, Any]:
    """Execute one ablation config and return a results dict."""
    set_seed(cfg.seed)
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"Run: axis={cfg.axis} value={cfg.axis_value} seed={cfg.seed}")
    print(f"  DP={'yes' if cfg.use_dp else 'no'} eps={cfg.target_eps} "
          f"batch={cfg.logical_batch_size}/{cfg.physical_batch_size} "
          f"grad_norm={cfg.max_grad_norm} augmult={cfg.augmult} "
          f"epochs={cfg.vae_epochs} latent={cfg.latent_dim} lr={cfg.vae_lr}")
    print(f"{'='*60}")

    # Load data
    real_by_label, cifar_train, test_loader = load_cifar_data(cfg)

    # Train VAE
    syn_x, syn_y, label_metrics = train_all_labels(real_by_label, cfg, device)

    # Aggregate training metrics
    mean_eps = float(np.mean([m["actual_epsilon"] for m in label_metrics]))
    mean_loss = float(np.mean([m["final_loss"] for m in label_metrics]))
    mean_noise = float(np.mean([m["noise_multiplier"] for m in label_metrics]))

    # Synthetic quality metrics
    quality = evaluate_quality(real_by_label, syn_x, syn_y, device)

    # Synthetic-only classifier accuracy
    acc_syn_only = train_and_eval_classifier(
        syn_x, syn_y, test_loader, device,
        cfg.classifier_epochs, cfg.classifier_lr, cfg.classifier_batch_size,
    )

    # Few-shot evaluation
    fewshot_results: Dict[str, float] = {}
    for i, n_local in enumerate(cfg.fewshot_per_class):
        local_x, local_y = sample_fewshot(cifar_train, n_local, cfg.seed + 100 + i)

        acc_real = train_and_eval_classifier(
            local_x, local_y, test_loader, device,
            cfg.classifier_epochs, cfg.classifier_lr, cfg.classifier_batch_size,
        )
        aug_x = torch.cat([local_x, syn_x])
        aug_y = torch.cat([local_y, syn_y])
        acc_aug = train_and_eval_classifier(
            aug_x, aug_y, test_loader, device,
            cfg.classifier_epochs, cfg.classifier_lr, cfg.classifier_batch_size,
        )
        fewshot_results[f"real_only_acc_{n_local}"] = acc_real
        fewshot_results[f"real_plus_syn_acc_{n_local}"] = acc_aug
        fewshot_results[f"delta_pp_{n_local}"] = (acc_aug - acc_real) * 100.0
        print(f"  fewshot n={n_local}: real={acc_real:.4f} +syn={acc_aug:.4f} "
              f"delta={fewshot_results[f'delta_pp_{n_local}']:.2f}pp")

    wall_time = time.time() - t0

    result = {
        "axis": cfg.axis,
        "axis_value": cfg.axis_value if cfg.axis_value is not None else "inf",
        "seed": cfg.seed,
        # DP params
        "target_eps": cfg.target_eps if cfg.target_eps is not None else float("inf"),
        "logical_batch_size": cfg.logical_batch_size,
        "physical_batch_size": cfg.physical_batch_size,
        "max_grad_norm": cfg.max_grad_norm,
        "augmult": cfg.augmult,
        "vae_epochs": cfg.vae_epochs,
        "latent_dim": cfg.latent_dim,
        "vae_lr": cfg.vae_lr,
        # Training metrics
        "mean_actual_epsilon": mean_eps,
        "mean_noise_multiplier": mean_noise,
        "mean_final_loss": mean_loss,
        "wall_time_s": wall_time,
        # Quality metrics
        "alpha_precision": quality["alpha_precision"],
        "beta_recall": quality["beta_recall"],
        "authenticity": quality["authenticity"],
        # Downstream accuracy
        "syn_only_acc": acc_syn_only,
        **fewshot_results,
    }

    # Log to W&B
    if wb_run is not None:
        wandb.log({f"ablation/{k}": v for k, v in result.items()
                   if isinstance(v, (int, float))})

    print(f"  quality: alpha_prec={quality['alpha_precision']:.3f} "
          f"beta_rec={quality['beta_recall']:.3f} "
          f"auth={quality['authenticity']:.3f}")
    print(f"  syn_only_acc={acc_syn_only:.4f} wall_time={wall_time:.1f}s")

    return result


# ---------------------------------------------------------------------------
# W&B init
# ---------------------------------------------------------------------------
def init_wandb(
    axis: str,
    disable: bool = False,
    offline: bool = False,
) -> Optional[wandb.sdk.wandb_run.Run]:
    if disable:
        return None
    if offline:
        os.environ["WANDB_MODE"] = "offline"
    try:
        run = wandb.init(
            project="cifar_ablation",
            name=f"ablation_{axis}",
            config={"axis": axis, "baseline": BASELINE, "seeds": SEEDS},
        )
        return run
    except Exception as exc:
        print(f"W&B init failed ({type(exc).__name__}): {exc}")
        return None


def _expected_axis_run_count(axis: str) -> int:
    return len(ABLATION_AXES[axis]) * len(SEEDS)


def _normalize_axis_value_for_key(v: Any) -> str:
    if v is None:
        return "inf"
    try:
        fv = float(v)
        if np.isinf(fv):
            return "inf"
        return f"{fv:g}"
    except Exception:
        return str(v)


def _cfg_progress_key(cfg: "RunConfig") -> Tuple[str, int]:
    return (_normalize_axis_value_for_key(cfg.axis_value), int(cfg.seed))


def get_wandb_axis_progress(entity: str, project: str) -> Dict[str, Dict[str, Any]]:
    """Return per-axis progress based on unique (axis_value, seed) rows in W&B history."""
    progress: Dict[str, Dict[str, Any]] = {
        axis: {
            "count": 0,
            "expected": _expected_axis_run_count(axis),
            "complete": False,
            "seen": set(),
            "runs": [],
        }
        for axis in ABLATION_AXES
    }
    try:
        api = wandb.Api(timeout=60)
        runs = api.runs(f"{entity}/{project}")
        history_keys = ["ablation/axis_value", "ablation/seed"] + [
            f"ablation/{param}" for param in AXIS_TO_PARAM.values()
        ]
        for run in runs:
            axis = (run.config or {}).get("axis")
            if not (isinstance(axis, str) and axis in ABLATION_AXES):
                continue

            progress[axis]["runs"].append({"id": run.id, "name": run.name, "state": run.state})

            # Count unique config points from history; handles partially completed runs.
            try:
                axis_param_key = f"ablation/{AXIS_TO_PARAM[axis]}"
                for item in run.scan_history(keys=history_keys):
                    seed = item.get("ablation/seed")
                    axis_value = item.get("ablation/axis_value")
                    if axis_value is None:
                        # Fallback for no-DP target_eps runs where axis_value is a non-numeric
                        # string ("inf") and therefore not logged by the numeric-only filter.
                        axis_value = item.get(axis_param_key)
                    if seed is None or axis_value is None:
                        continue
                    try:
                        seed_key = int(round(float(seed)))
                    except Exception:
                        continue
                    progress[axis]["seen"].add((_normalize_axis_value_for_key(axis_value), seed_key))
            except Exception as exc:
                print(f"Warning: failed to scan W&B history for run {run.id} ({type(exc).__name__}): {exc}")
    except Exception as exc:
        print(f"Warning: failed to query W&B runs ({type(exc).__name__}): {exc}")
        return progress

    for axis, info in progress.items():
        info["count"] = len(info["seen"])
        info["complete"] = info["count"] >= info["expected"]
    return progress


# ---------------------------------------------------------------------------
# Main local execution
# ---------------------------------------------------------------------------
def run_local(
    axes: Optional[List[str]],
    dry_run: bool,
    disable_wandb: bool,
    offline_wandb: bool,
    wandb_progress: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    configs = build_grid(axes)

    if wandb_progress is not None:
        before = len(configs)
        filtered: List[RunConfig] = []
        skipped = 0
        for cfg in configs:
            seen = wandb_progress.get(cfg.axis, {}).get("seen", set())
            if _cfg_progress_key(cfg) in seen:
                skipped += 1
                continue
            filtered.append(cfg)
        configs = filtered
        if skipped:
            print(f"Resume mode: skipping {skipped} completed runs already found in W&B history")

    print(f"Ablation grid: {len(configs)} runs")

    if dry_run:
        print("\n--- DRY RUN: Grid summary ---")
        for ax in (axes or list(ABLATION_AXES.keys())):
            ax_configs = [c for c in configs if c.axis == ax]
            values = sorted(set(
                c.axis_value if c.axis_value is not None else "inf"
                for c in ax_configs
            ), key=str)
            print(f"  {ax}: {len(ax_configs)} runs ({len(ax_configs)//len(SEEDS)} values × {len(SEEDS)} seeds)")
            print(f"    values: {values}")
        total = len(configs)
        axes_used = axes or list(ABLATION_AXES.keys())
        n_unique = sum(
            len(set(
                c.axis_value if c.axis_value is not None else "inf"
                for c in configs if c.axis == a
            ))
            for a in axes_used
        )
        print(f"\nTotal remaining: {n_unique} unique configs across {len(axes_used)} axes = {total} runs")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results_dir = REPO_ROOT / "results" / "cifar_ablation"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []

    # Group configs by axis for W&B grouping
    axes_in_grid = sorted(set(c.axis for c in configs))
    for axis in axes_in_grid:
        axis_configs = [c for c in configs if c.axis == axis]
        print(f"\n{'#'*60}")
        print(f"# AXIS: {axis} ({len(axis_configs)} runs)")
        print(f"{'#'*60}")

        wb_run = init_wandb(axis, disable=disable_wandb, offline=offline_wandb)
        axis_results: List[Dict[str, Any]] = []

        try:
            for cfg in axis_configs:
                result = run_single(cfg, device, wb_run)
                axis_results.append(result)
                all_results.append(result)
                # Free Python and CUDA cached memory between configs.
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
        finally:
            if wb_run is not None:
                wandb.finish()

        # Save per-axis CSV
        axis_df = pd.DataFrame(axis_results)
        axis_csv = results_dir / f"ablation_{axis}.csv"
        axis_df.to_csv(axis_csv, index=False)
        print(f"\nSaved {axis_csv} ({len(axis_results)} rows)")

    # Save combined CSV
    all_df = pd.DataFrame(all_results)
    combined_csv = results_dir / "ablation_all.csv"
    all_df.to_csv(combined_csv, index=False)
    print(f"\nSaved {combined_csv} ({len(all_results)} rows)")

    # Save summary JSON (best config per axis)
    summary: Dict[str, Any] = {}
    for axis in axes_in_grid:
        axis_df = all_df[all_df["axis"] == axis]
        # Average over seeds
        grouped = axis_df.groupby("axis_value").agg({
            "alpha_precision": "mean",
            "beta_recall": "mean",
            "authenticity": "mean",
            "syn_only_acc": "mean",
            "mean_actual_epsilon": "mean",
            "mean_final_loss": "mean",
            "wall_time_s": "mean",
        }).reset_index()

        # Best by alpha_precision
        best_idx = grouped["alpha_precision"].idxmax()
        best_row = grouped.iloc[best_idx]
        summary[axis] = {
            "best_value": str(best_row["axis_value"]),
            "best_alpha_precision": float(best_row["alpha_precision"]),
            "best_beta_recall": float(best_row["beta_recall"]),
            "best_authenticity": float(best_row["authenticity"]),
            "best_syn_only_acc": float(best_row["syn_only_acc"]),
        }

    summary_path = results_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")


# ---------------------------------------------------------------------------
# HPC script generation
# ---------------------------------------------------------------------------
def generate_lsf_script(
    axis: Optional[str] = None,
    resume_missing_wandb: bool = False,
    wandb_entity: str = "rohwedertimm",
    wandb_project: str = "cifar_ablation",
) -> str:
    script_rel = Path(__file__).resolve().relative_to(REPO_ROOT)
    hpc = HPC_DEFAULTS
    extra_args = ""
    if resume_missing_wandb:
        extra_args = (
            f" --resume-missing-wandb"
            f" --wandb-entity {wandb_entity}"
            f" --wandb-project {wandb_project}"
        )
    axes_arg = f" --axes {axis}" if axis else ""
    job_suffix = axis if axis else "all"
    return f"""#!/bin/bash
#BSUB -J cifar-ablation-{job_suffix}
#BSUB -q {hpc['hpc_queue']}
#BSUB -W {hpc['hpc_walltime']}
#BSUB -n {hpc['hpc_cores']}
#BSUB -R "rusage[mem={hpc['hpc_mem']}]"
#BSUB -M {hpc['hpc_mem']}
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num={hpc['hpc_gpus']}:mode=exclusive_process"
#BSUB -o logs/cifar_ablation_{job_suffix}_%J.out
#BSUB -e logs/cifar_ablation_{job_suffix}_%J.err

module load python3/3.12.4
module load cuda/12.1

cd {hpc['hpc_project_dir']}
mkdir -p logs

if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "ERROR: no virtual environment found (venv or .venv)"
  exit 1
fi

export TF_CPP_MIN_LOG_LEVEL=3
export GRPC_VERBOSITY=ERROR
export WANDB_SILENT=true
export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::FutureWarning,ignore::UserWarning"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

if [ -n "$WANDB_API_KEY" ]; then
  export WANDB_MODE=online
else
  export WANDB_MODE=offline
fi

python {script_rel} --execution-mode local{axes_arg}{extra_args}
"""


def submit_to_hpc(
    axes: Optional[List[str]],
    resume_missing_wandb: bool = False,
    wandb_entity: str = "rohwedertimm",
    wandb_project: str = "cifar_ablation",
    single_script: bool = False,
) -> None:
    if axes is None:
        axes = list(ABLATION_AXES.keys())

    hpc_dir = REPO_ROOT / "hpc"
    hpc_dir.mkdir(parents=True, exist_ok=True)

    if single_script:
        axes_arg = ",".join(axes)
        script_content = generate_lsf_script(
            axes_arg,
            resume_missing_wandb=resume_missing_wandb,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
        )
        script_path = hpc_dir / "cifar_ablation_all_hpc.sh"
        script_path.write_text(script_content, newline="\n")
        print(f"Generated combined script: {script_path}")
        try:
            remote_dir = f"{HPC_DEFAULTS['hpc_project_dir']}/hpc/"
            subprocess.run(
                ["scp", str(script_path), f"{HPC_DEFAULTS['hpc_host']}:{remote_dir}"],
                check=True,
            )
            print("  Copied to HPC. Submit with:")
            print(f"  bsub < hpc/{script_path.name}")
        except subprocess.CalledProcessError as exc:
            print(f"  SCP failed: {exc}")
            print(f"  Script saved locally at: {script_path}")
            print("Submit locally after manual copy with:")
            print(f"  bsub < hpc/{script_path.name}")
        return

    for axis in axes:
        script_content = generate_lsf_script(
            axis,
            resume_missing_wandb=resume_missing_wandb,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
        )
        script_path = hpc_dir / f"cifar_ablation_{axis}_hpc.sh"
        script_path.write_text(script_content, newline="\n")
        print(f"Generated: {script_path}")

        try:
            remote_dir = f"{HPC_DEFAULTS['hpc_project_dir']}/hpc/"
            subprocess.run(
                ["scp", str(script_path), f"{HPC_DEFAULTS['hpc_host']}:{remote_dir}"],
                check=True,
            )
            print(f"  Copied to HPC. Submit with:")
            print(f"  bsub < hpc/{script_path.name}")
        except subprocess.CalledProcessError as exc:
            print(f"  SCP failed: {exc}")
            print(f"  Script saved locally at: {script_path}")

    print(f"\nTo submit all axes at once:")
    print(f"  ssh {HPC_DEFAULTS['hpc_host']}")
    print(f"  cd {HPC_DEFAULTS['hpc_project_dir']}")
    for axis in axes:
        print(f"  bsub < hpc/cifar_ablation_{axis}_hpc.sh")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="CIFAR DP++ Ablation Study")
    parser.add_argument("--execution-mode", choices=["local", "hpc"], default="local")
    parser.add_argument("--axes", type=str, default=None,
                        help="Comma-separated axes to run (e.g. 'epsilon,augmult'). "
                             "Default: all axes.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print grid without running.")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--offline-wandb", action="store_true")
    parser.add_argument("--resume-missing-wandb", action="store_true",
                        help="Query W&B and run only axes that do not yet have a finished run.")
    parser.add_argument("--wandb-entity", type=str, default="rohwedertimm",
                        help="W&B entity/user for --resume-missing-wandb.")
    parser.add_argument("--wandb-project", type=str, default="cifar_ablation",
                        help="W&B project for --resume-missing-wandb.")
    parser.add_argument("--hpc-single-script", action="store_true",
                        help="With --execution-mode hpc, generate one combined script for all selected axes.")
    args = parser.parse_args()

    # Normalize axis names: allow friendly names
    axis_aliases = {
        "epsilon": "target_eps",
        "eps": "target_eps",
        "batch_size": "logical_batch_size",
        "batch": "logical_batch_size",
        "physical_batch": "physical_batch_size",
        "grad_norm": "max_grad_norm",
        "clip": "max_grad_norm",
        "epochs": "vae_epochs",
        "latent": "latent_dim",
        "lr": "vae_lr",
    }

    axes = None
    if args.axes:
        raw_axes = [a.strip() for a in args.axes.split(",")]
        axes = [axis_aliases.get(a, a) for a in raw_axes]

    wandb_progress = None
    if args.resume_missing_wandb:
        requested_axes = axes or list(ABLATION_AXES.keys())
        wandb_progress = get_wandb_axis_progress(args.wandb_entity, args.wandb_project)
        completed = [a for a in requested_axes if wandb_progress.get(a, {}).get("complete")]
        missing = [a for a in requested_axes if a not in completed]
        print(f"W&B axis progress in {args.wandb_entity}/{args.wandb_project}:")
        for a in requested_axes:
            p = wandb_progress.get(a, {})
            print(f"  {a}: {p.get('count', 0)}/{p.get('expected', _expected_axis_run_count(a))} "
                  f"({'complete' if p.get('complete') else 'incomplete'})")
        print(f"Selected missing axes: {missing}")
        axes = missing
        if not axes:
            print("Nothing to run: all selected axes already present in W&B.")
            return

    print("CIFAR DP++ Ablation Study")
    print("=" * 40)

    if args.execution_mode == "hpc":
        submit_to_hpc(
            axes,
            resume_missing_wandb=args.resume_missing_wandb,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            single_script=args.hpc_single_script,
        )
    else:
        run_local(axes, args.dry_run, args.disable_wandb, args.offline_wandb, wandb_progress=wandb_progress)


if __name__ == "__main__":
    main()
