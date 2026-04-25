#!/usr/bin/env python3
"""
Run CIFAR DP++ synthetic-data experiment from notebook as a script.

Mirrors:
  notebooks/cifar/cifar_saliency_xai_dpddpm_style.ipynb

Supports:
  - local execution
  - HPC script generation/copy (DTU LSF style)
  - W&B logging of metrics, tables, and figures
"""

from __future__ import annotations

import argparse
import math
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


CONFIG = {
    "execution_mode": "local",  # "local" | "hpc"
    "dry_run": False,
    # HPC (override via .env; see .env.example)
    "hpc_host": os.environ.get("HPC_HOST", "user@hpc.example.com"),
    "hpc_project_dir": os.environ.get("HPC_PROJECT_DIR", "~/dp-fedaug"),
    "hpc_queue": "gpua100",
    "hpc_walltime": "24:00",
    "hpc_gpus": 1,
    # DTU GPU queues require at least 4 CPU cores per GPU.
    "hpc_cores": 4,
    # LSF memory is per slot/core. Keep this aligned with -n.
    "hpc_total_mem_gb": 100,
    "hpc_script_path": "hpc/cifar_saliency_dpddpm_style_hpc.sh",
    # Data/paths
    "seed": 42,
    "data_dir": "data",
    "output_dir": "visual/outputs",
    "results_dir": "results/cifar_saliency",
    # DP++ VAE
    "n_real_per_class": 2000,
    "target_eps": 1.0,
    "delta": 1e-5,
    "vae_epochs": 40,
    "latent_dim": 96,
    "vae_lr": 1e-3,
    "logical_batch_size": 512,
    "physical_batch_size": 64,
    "augmult": 4,
    "max_grad_norm": 1.0,
    "synthetic_count_per_class": 200,
    # Downstream utility
    "classifier_epochs": 15,
    "classifier_lr": 1e-3,
    "classifier_batch_size": 64,
    "fewshot_per_class": [10, 50, 100],
    "test_batch_size": 256,
    # W&B
    "wandb_project": "cifar_saliency",
    "wandb_entity": None,
    "wandb_run_name": "cifar_dpplusplus_style",
    "offline_wandb": False,
    "disable_wandb": False,
}


@dataclass
class Config:
    execution_mode: str
    dry_run: bool
    hpc_host: str
    hpc_project_dir: str
    hpc_queue: str
    hpc_walltime: str
    hpc_gpus: int
    hpc_cores: int
    hpc_total_mem_gb: int
    hpc_script_path: Path
    seed: int
    data_dir: Path
    output_dir: Path
    results_dir: Path
    n_real_per_class: int
    target_eps: float
    delta: float
    vae_epochs: int
    latent_dim: int
    vae_lr: float
    logical_batch_size: int
    physical_batch_size: int
    augmult: int
    max_grad_norm: float
    synthetic_count_per_class: int
    classifier_epochs: int
    classifier_lr: float
    classifier_batch_size: int
    fewshot_per_class: List[int]
    test_batch_size: int
    wandb_project: str
    wandb_entity: Optional[str]
    wandb_run_name: str
    offline_wandb: bool
    disable_wandb: bool


def build_config(execution_mode_override: Optional[str]) -> Config:
    raw = dict(CONFIG)
    if execution_mode_override:
        raw["execution_mode"] = execution_mode_override
    return Config(
        execution_mode=str(raw["execution_mode"]),
        dry_run=bool(raw["dry_run"]),
        hpc_host=str(raw["hpc_host"]),
        hpc_project_dir=str(raw["hpc_project_dir"]),
        hpc_queue=str(raw["hpc_queue"]),
        hpc_walltime=str(raw["hpc_walltime"]),
        hpc_gpus=int(raw["hpc_gpus"]),
        hpc_cores=int(raw["hpc_cores"]),
        hpc_total_mem_gb=int(raw["hpc_total_mem_gb"]),
        hpc_script_path=REPO_ROOT / str(raw["hpc_script_path"]),
        seed=int(raw["seed"]),
        data_dir=REPO_ROOT / str(raw["data_dir"]),
        output_dir=REPO_ROOT / str(raw["output_dir"]),
        results_dir=REPO_ROOT / str(raw["results_dir"]),
        n_real_per_class=int(raw["n_real_per_class"]),
        target_eps=float(raw["target_eps"]),
        delta=float(raw["delta"]),
        vae_epochs=int(raw["vae_epochs"]),
        latent_dim=int(raw["latent_dim"]),
        vae_lr=float(raw["vae_lr"]),
        logical_batch_size=int(raw["logical_batch_size"]),
        physical_batch_size=int(raw["physical_batch_size"]),
        augmult=int(raw["augmult"]),
        max_grad_norm=float(raw["max_grad_norm"]),
        synthetic_count_per_class=int(raw["synthetic_count_per_class"]),
        classifier_epochs=int(raw["classifier_epochs"]),
        classifier_lr=float(raw["classifier_lr"]),
        classifier_batch_size=int(raw["classifier_batch_size"]),
        fewshot_per_class=[int(x) for x in raw["fewshot_per_class"]],
        test_batch_size=int(raw["test_batch_size"]),
        wandb_project=str(raw["wandb_project"]),
        wandb_entity=raw["wandb_entity"],
        wandb_run_name=str(raw["wandb_run_name"]),
        offline_wandb=bool(raw["offline_wandb"]),
        disable_wandb=bool(raw["disable_wandb"]),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cifar_by_label(
    cfg: Config, seed: int
) -> Tuple[Dict[int, torch.Tensor], datasets.CIFAR10, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()])
    cifar_train = datasets.CIFAR10(root=cfg.data_dir, train=True, download=True, transform=transform)
    cifar_test = datasets.CIFAR10(root=cfg.data_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(cifar_test, batch_size=cfg.test_batch_size, shuffle=False)

    targets = np.array(cifar_train.targets)
    rng = np.random.default_rng(seed)

    real_by_label: Dict[int, torch.Tensor] = {}
    for lbl in range(10):
        indices = np.where(targets == lbl)[0]
        if cfg.n_real_per_class > len(indices):
            raise ValueError(
                f"Requested n_real_per_class={cfg.n_real_per_class} for label={lbl}, "
                f"but only {len(indices)} samples available."
            )
        chosen = rng.choice(indices, size=cfg.n_real_per_class, replace=False)
        images = torch.stack([cifar_train[int(i)][0] for i in chosen], dim=0)
        real_by_label[lbl] = images

    return real_by_label, cifar_train, test_loader


def train_cifar_vae_dp_plus(
    real_by_label: Dict[int, torch.Tensor],
    cfg: Config,
    device: torch.device,
    wb_run: Optional[wandb.sdk.wandb_run.Run],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, float], List[Dict[str, float]]]:
    syn_images: List[torch.Tensor] = []
    syn_labels: List[torch.Tensor] = []
    eps_per_label: Dict[int, float] = {}
    label_train_rows: List[Dict[str, float]] = []

    for lbl in range(10):
        subset = real_by_label[lbl]
        n = subset.shape[0]
        sample_rate = min(cfg.logical_batch_size / max(n, 1), 0.99)
        noise_multiplier = float(
            get_noise_multiplier(
                target_epsilon=float(cfg.target_eps),
                target_delta=cfg.delta,
                sample_rate=sample_rate,
                epochs=cfg.vae_epochs,
            )
        )
        print(
            f"[DP++] label={lbl} n={n} eps={cfg.target_eps} "
            f"sample_rate={sample_rate:.4f} noise_multiplier={noise_multiplier:.4f}"
        )

        ds = TensorDataset(subset)
        dl = DataLoader(ds, batch_size=cfg.logical_batch_size, shuffle=True, drop_last=False)
        model = CIFAR_VAE_PP(latent_dim=cfg.latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.vae_lr)
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
            with BatchMemoryManager(
                data_loader=dl,
                max_physical_batch_size=cfg.physical_batch_size,
                optimizer=optimizer,
            ) as mem_dl:
                for (xb,) in mem_dl:
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
            if wb_run is not None:
                wandb.log(
                    {
                        "train/label": lbl,
                        "train/epoch": epoch + 1,
                        "train/loss": last_loss,
                        "train/kl_weight": kl_weight,
                        "train/sample_rate": sample_rate,
                        "train/noise_multiplier": noise_multiplier,
                    }
                )

        eps = float(pe.get_epsilon(cfg.delta))
        eps_per_label[lbl] = eps

        model.eval()
        decoder_fn = model.decode if hasattr(model, "decode") else model._module.decode
        with torch.no_grad():
            z = torch.randn(cfg.synthetic_count_per_class, cfg.latent_dim, device=device)
            syn = decoder_fn(z).cpu().clamp(0, 1)

        syn_images.append(syn)
        syn_labels.append(torch.full((cfg.synthetic_count_per_class,), lbl, dtype=torch.long))

        row = {
            "label": lbl,
            "actual_epsilon": eps,
            "noise_multiplier": noise_multiplier,
            "sample_rate": sample_rate,
            "final_epoch_loss": last_loss,
            "n_real_per_class": n,
        }
        label_train_rows.append(row)
        if wb_run is not None:
            wandb.log({f"dp_label/{k}": v for k, v in row.items() if isinstance(v, (int, float))})

    return torch.cat(syn_images, dim=0), torch.cat(syn_labels, dim=0), eps_per_label, label_train_rows


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
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += int((pred == yb).sum().item())
            total += int(yb.size(0))
    return correct / max(total, 1)


def sample_fewshot_from_dataset(
    cifar_train: datasets.CIFAR10,
    n_per_class: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    targets = np.array(cifar_train.targets)
    rng = np.random.default_rng(seed)
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    for lbl in range(10):
        indices = np.where(targets == lbl)[0]
        if n_per_class > len(indices):
            raise ValueError(f"Requested n_per_class={n_per_class} for label={lbl}, max={len(indices)}")
        chosen = rng.choice(indices, size=n_per_class, replace=False)
        x = torch.stack([cifar_train[int(i)][0] for i in chosen], dim=0)
        y = torch.full((n_per_class,), lbl, dtype=torch.long)
        xs.append(x)
        ys.append(y)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def create_figures(
    real_by_label: Dict[int, torch.Tensor],
    syn_x: torch.Tensor,
    syn_y: torch.Tensor,
    fewshot_df: pd.DataFrame,
    cfg: Config,
) -> Tuple[Path, Path]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = cfg.output_dir / "cifar_saliency_dpddpm_samples.png"
    utility_path = cfg.output_dir / "cifar_saliency_dpddpm_downstream.png"

    fig, axes = plt.subplots(2, 10, figsize=(18, 4))
    for cls in range(10):
        img_real = real_by_label[cls][0].permute(1, 2, 0).numpy()
        img_syn = syn_x[syn_y == cls][0].permute(1, 2, 0).numpy()
        axes[0, cls].imshow(np.clip(img_real, 0, 1))
        axes[0, cls].axis("off")
        axes[0, cls].set_title(CLASS_NAMES[cls], fontsize=9)
        axes[1, cls].imshow(np.clip(img_syn, 0, 1))
        axes[1, cls].axis("off")
    axes[0, 0].set_ylabel("Real", fontsize=11)
    axes[1, 0].set_ylabel("DP++ Syn", fontsize=11)
    plt.tight_layout()
    plt.savefig(samples_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [f"{int(n)}/cls" for n in fewshot_df["n_real_per_class"].tolist()]
    real_vals = fewshot_df["real_only_acc"].tolist()
    aug_vals = fewshot_df["real_plus_syn_acc"].tolist()
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, real_vals, width=w, label="Real-only", color="#333333")
    ax.bar(x + w / 2, aug_vals, width=w, label="Real + DP++ synthetic", color="#2E7D32")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Downstream utility with DP++ synthetic augmentation")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(utility_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return samples_path, utility_path


def init_wandb(cfg: Config) -> Optional[wandb.sdk.wandb_run.Run]:
    if cfg.disable_wandb:
        return None
    if cfg.offline_wandb:
        os.environ["WANDB_MODE"] = "offline"
    try:
        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            config={k: v for k, v in CONFIG.items()},
        )
        return run
    except Exception as exc:
        print(f"W&B init failed ({type(exc).__name__}): {exc}")
        print("Continuing without W&B logging.")
        return None


def run_local(cfg: Config) -> None:
    set_seed(cfg.seed)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    wb_run = init_wandb(cfg)
    try:
        real_by_label, cifar_train, test_loader = load_cifar_by_label(cfg, seed=cfg.seed)
        syn_x, syn_y, eps_per_label, label_train_rows = train_cifar_vae_dp_plus(
            real_by_label=real_by_label,
            cfg=cfg,
            device=device,
            wb_run=wb_run,
        )
        print(f"Synthetic shape: x={tuple(syn_x.shape)} y={tuple(syn_y.shape)}")
        print(f"Per-label epsilon: { {k: round(v, 3) for k, v in eps_per_label.items()} }")

        acc_syn = train_and_eval_classifier(
            syn_x,
            syn_y,
            test_loader=test_loader,
            device=device,
            epochs=cfg.classifier_epochs,
            lr=cfg.classifier_lr,
            batch_size=cfg.classifier_batch_size,
        )
        print(f"Synthetic-only accuracy (DP++): {acc_syn:.4f}")

        fewshot_rows: List[Dict[str, float]] = []
        for i, n_local in enumerate(cfg.fewshot_per_class):
            local_x, local_y = sample_fewshot_from_dataset(
                cifar_train=cifar_train,
                n_per_class=n_local,
                seed=cfg.seed + 100 + i,
            )
            acc_real = train_and_eval_classifier(
                local_x,
                local_y,
                test_loader=test_loader,
                device=device,
                epochs=cfg.classifier_epochs,
                lr=cfg.classifier_lr,
                batch_size=cfg.classifier_batch_size,
            )
            aug_x = torch.cat([local_x, syn_x], dim=0)
            aug_y = torch.cat([local_y, syn_y], dim=0)
            acc_aug = train_and_eval_classifier(
                aug_x,
                aug_y,
                test_loader=test_loader,
                device=device,
                epochs=cfg.classifier_epochs,
                lr=cfg.classifier_lr,
                batch_size=cfg.classifier_batch_size,
            )
            delta_pp = (acc_aug - acc_real) * 100.0
            print(
                f"n_real={n_local}/class -> real-only={acc_real:.4f}, "
                f"+DP++syn={acc_aug:.4f}, delta={delta_pp:.2f}pp"
            )
            row = {
                "n_real_per_class": int(n_local),
                "real_only_acc": float(acc_real),
                "real_plus_syn_acc": float(acc_aug),
                "delta_pp": float(delta_pp),
            }
            fewshot_rows.append(row)
            if wb_run is not None:
                wandb.log({f"fewshot/{k}": v for k, v in row.items()})

        fewshot_df = pd.DataFrame(fewshot_rows)
        eps_df = pd.DataFrame(
            [{"label": lbl, "actual_epsilon": float(eps)} for lbl, eps in sorted(eps_per_label.items())]
        )
        train_df = pd.DataFrame(label_train_rows)

        samples_path, utility_path = create_figures(real_by_label, syn_x, syn_y, fewshot_df, cfg)

        summary = {
            "synthetic_only_accuracy": float(acc_syn),
            "mean_label_epsilon": float(np.mean(list(eps_per_label.values()))),
            "max_label_epsilon": float(np.max(list(eps_per_label.values()))),
            "min_label_epsilon": float(np.min(list(eps_per_label.values()))),
            "best_fewshot_aug_acc": float(fewshot_df["real_plus_syn_acc"].max()),
            "best_fewshot_delta_pp": float(fewshot_df["delta_pp"].max()),
        }

        fewshot_csv = cfg.results_dir / "fewshot_results.csv"
        eps_csv = cfg.results_dir / "epsilon_per_label.csv"
        train_csv = cfg.results_dir / "dp_label_training_metrics.csv"
        summary_json = cfg.results_dir / "summary.json"
        fewshot_df.to_csv(fewshot_csv, index=False)
        eps_df.to_csv(eps_csv, index=False)
        train_df.to_csv(train_csv, index=False)
        summary_json.write_text(json.dumps(summary, indent=2))
        print(f"Saved results to {cfg.results_dir}")

        if wb_run is not None:
            wandb.log(
                {
                    "metrics/synthetic_only_accuracy": summary["synthetic_only_accuracy"],
                    "metrics/mean_label_epsilon": summary["mean_label_epsilon"],
                    "metrics/max_label_epsilon": summary["max_label_epsilon"],
                    "metrics/min_label_epsilon": summary["min_label_epsilon"],
                    "metrics/best_fewshot_aug_acc": summary["best_fewshot_aug_acc"],
                    "metrics/best_fewshot_delta_pp": summary["best_fewshot_delta_pp"],
                    "tables/fewshot": wandb.Table(dataframe=fewshot_df),
                    "tables/epsilon_per_label": wandb.Table(dataframe=eps_df),
                    "tables/dp_label_training_metrics": wandb.Table(dataframe=train_df),
                    "figures/samples_grid": wandb.Image(str(samples_path)),
                    "figures/downstream_utility": wandb.Image(str(utility_path)),
                }
            )
            wandb.run.summary.update(summary)

            artifact = wandb.Artifact("cifar_saliency_outputs", type="dataset")
            artifact.add_file(str(fewshot_csv))
            artifact.add_file(str(eps_csv))
            artifact.add_file(str(train_csv))
            artifact.add_file(str(summary_json))
            artifact.add_file(str(samples_path))
            artifact.add_file(str(utility_path))
            wandb.log_artifact(artifact)
    finally:
        if wb_run is not None:
            wandb.finish()


def generate_lsf_script(cfg: Config) -> str:
    script_rel = Path(__file__).resolve().relative_to(REPO_ROOT)
    mem_per_slot_gb = max(1, math.ceil(cfg.hpc_total_mem_gb / max(cfg.hpc_cores, 1)))
    return f"""#!/bin/bash
#BSUB -J cifar-saliency-dpddpm
#BSUB -q {cfg.hpc_queue}
#BSUB -W {cfg.hpc_walltime}
#BSUB -n {cfg.hpc_cores}
#BSUB -R "rusage[mem={mem_per_slot_gb}GB]"
#BSUB -M {mem_per_slot_gb}GB
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num={cfg.hpc_gpus}:mode=exclusive_process"
#BSUB -o logs/cifar_saliency_dpddpm_%J.out
#BSUB -e logs/cifar_saliency_dpddpm_%J.err

module load python3/3.12.4
module load cuda/12.1

cd {cfg.hpc_project_dir}
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

python {script_rel} --execution-mode local
"""


def submit_to_hpc(cfg: Config) -> None:
    lsf_script = generate_lsf_script(cfg)
    cfg.hpc_script_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.hpc_script_path.write_text(lsf_script, newline="\n")
    print(f"Generated LSF script: {cfg.hpc_script_path}")

    try:
        remote_dir = f"{cfg.hpc_project_dir}/hpc/"
        subprocess.run(
            ["scp", str(cfg.hpc_script_path), f"{cfg.hpc_host}:{remote_dir}"],
            check=True,
        )
        print("Copied script to HPC successfully.")
        print("Submit with:")
        print(f"  ssh {cfg.hpc_host}")
        print(f"  cd {cfg.hpc_project_dir}")
        print(f"  bsub < {cfg.hpc_script_path.relative_to(REPO_ROOT)}")
    except subprocess.CalledProcessError as exc:
        print(f"Error copying to HPC: {exc}")
        print(f"Script was generated locally at: {cfg.hpc_script_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CIFAR DP++ notebook-to-script runner")
    parser.add_argument("--execution-mode", choices=["local", "hpc"], default=None)
    args = parser.parse_args()

    cfg = build_config(execution_mode_override=args.execution_mode)
    print("CIFAR DP++ Script Runner")
    print("=" * 40)
    print(f"execution_mode={cfg.execution_mode}")
    print(f"wandb_project={cfg.wandb_project}")
    print(f"seed={cfg.seed}")
    print(f"hpc_queue={cfg.hpc_queue}")
    print(f"hpc_cores={cfg.hpc_cores}")
    print(f"hpc_total_mem_gb={cfg.hpc_total_mem_gb}")
    print(f"n_real_per_class={cfg.n_real_per_class}")
    print(f"target_eps={cfg.target_eps}")
    print(f"synthetic_count_per_class={cfg.synthetic_count_per_class}")
    print(f"fewshot_per_class={cfg.fewshot_per_class}")
    print(f"output_dir={cfg.output_dir}")
    print(f"results_dir={cfg.results_dir}")

    if cfg.dry_run:
        return

    if cfg.execution_mode == "hpc":
        submit_to_hpc(cfg)
    else:
        run_local(cfg)


if __name__ == "__main__":
    main()
