"""
metrics_v2.py – α‑Precision, β‑Recall, Authenticity + Auditing helper

Works with torchvision 0.12 → 0.21.
"""

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from sklearn.neighbors import NearestNeighbors
from typing import Literal, Union, Dict, Tuple

# --------------------------------- weights compatibility (torchvision ≥0.15) --
try:
    from torchvision.models import Inception_V3_Weights, ResNet18_Weights
except ImportError:
    Inception_V3_Weights = ResNet18_Weights = None

# ──────────────────────────────────────────────────────────────────────────────
# helper functions
# ──────────────────────────────────────────────────────────────────────────────
def _default_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

def _load_backbone(name: Literal["inception", "resnet18"], device):
    if name == "inception":
        if Inception_V3_Weights:
            net = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
        else:
            net = models.inception_v3(pretrained=True, aux_logits=False)
        net.fc = torch.nn.Identity()
        dim = 2048
    elif name == "resnet18":
        if ResNet18_Weights:
            net = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            net = models.resnet18(pretrained=True)
        net.fc = torch.nn.Identity()
        dim = 512
    else:
        raise ValueError(name)
    net.eval().to(device)
    return net, dim

@torch.no_grad()
def _embed(data: Union[str, Dataset],
           model: torch.nn.Module,
           batch: int,
           device,
           img_size: int) -> np.ndarray:
    if isinstance(data, str):
        dataset = datasets.ImageFolder(data, _default_transform(img_size))
    elif isinstance(data, Dataset):
        dataset = data
    else:
        raise TypeError("data must be path or Dataset")

    # Handle empty datasets
    if len(dataset) == 0:
        # Return empty array with correct feature dimension (2048 for Inception V3)
        return np.empty((0, 2048), dtype=np.float32)

    loader = DataLoader(dataset, batch_size=batch, shuffle=False,
                        num_workers=0, pin_memory=True)
    feats = []
    for x, _ in loader:
        feats.append(model(x.to(device, non_blocking=True)).cpu())
    return torch.cat(feats).numpy() if feats else np.empty((0, 2048), dtype=np.float32)

# ──────────────────────────────────────────────────────────────────────────────
# public API – metrics
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_fidelity_diversity(
    real_source: Union[str, Dataset],
    fake_source: Union[str, Dataset],
    alpha: float = 0.90,
    beta: float = 0.90,
    batch_size: int = 64,
    device=None,
    backbone: Literal["inception", "resnet18"] = "inception",
    input_size: int | None = None
) -> Dict[str, float]:
    """
    Evaluate synthetic data quality using α-Precision, β-Recall, and Authenticity.

    Args:
        real_source: Path to real images folder or Dataset
        fake_source: Path to fake images folder or Dataset
        alpha: Quantile threshold for precision (default: 0.90)
        beta: Quantile threshold for recall (default: 0.90)
        batch_size: Batch size for feature extraction (default: 64)
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
        backbone: Feature extractor to use ('inception' or 'resnet18')
        input_size: Input image size (default: 299 for inception, 224 for resnet18)

    Returns:
        Dictionary with 'alpha_precision', 'beta_recall', and 'authenticity' scores
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if input_size is None:
        input_size = 299 if backbone == "inception" else 224

    net, _ = _load_backbone(backbone, device)
    real_f = _embed(real_source, net, batch_size, device, input_size)
    fake_f = _embed(fake_source, net, batch_size, device, input_size)

    # Handle edge case: too few real or fake samples for metrics
    if len(real_f) < 2 or len(fake_f) == 0:
        # Return default values when we can't compute meaningful metrics
        return dict(alpha_precision=1.0,
                    beta_recall=1.0,
                    authenticity=1.0)

    # α‑Precision --------------------------------------------------------------
    c_r = real_f.mean(0, keepdims=True)
    r_thr = np.quantile(np.linalg.norm(real_f - c_r, axis=1), alpha)
    alpha_prec = float((np.linalg.norm(fake_f - c_r, axis=1) <= r_thr).mean())

    # β‑Recall -----------------------------------------------------------------
    c_g = fake_f.mean(0, keepdims=True)
    g_thr = np.quantile(np.linalg.norm(fake_f - c_g, axis=1), beta)
    beta_rec = float((np.linalg.norm(real_f - c_g, axis=1) <= g_thr).mean())

    # Authenticity -------------------------------------------------------------
    nn_r2 = NearestNeighbors(n_neighbors=2).fit(real_f)
    d_r2 = nn_r2.kneighbors(real_f, return_distance=True)[0][:, 1]

    nn_fg = NearestNeighbors(n_neighbors=1).fit(real_f)
    d_fg, idx_fg = nn_fg.kneighbors(fake_f, return_distance=True)
    unauth_mask = (d_fg.flatten() <= d_r2[idx_fg.flatten()])
    authenticity = float(1.0 - unauth_mask.mean())

    return dict(alpha_precision=alpha_prec,
                beta_recall=beta_rec,
                authenticity=authenticity)

# ──────────────────────────────────────────────────────────────────────────────
# public API – auditing
# ──────────────────────────────────────────────────────────────────────────────
def audit_synthetic(
    real_source: Union[str, Dataset],
    fake_source: Union[str, Dataset, torch.Tensor],
    alpha: float = 0.90,
    beta: float = 0.90,
    batch_size: int = 64,
    device=None,
    backbone: Literal["inception", "resnet18"] = "inception",
    input_size: int | None = None,
    rule: Literal["precision_and_auth", "precision_only", "auth_only"] = "precision_and_auth"
) -> Dict[str, object]:
    """
    Post‑hoc sample‑level auditing (Sec. 5 of the paper):
      • computes precision + authenticity indicator per synthetic sample
      • filters by the chosen rule
      • returns masks & metrics before/after

    Args:
        real_source: Path to real images folder or Dataset
        fake_source: Path to fake images folder, Dataset, or Tensor
        alpha: Quantile threshold for precision (default: 0.90)
        beta: Quantile threshold for recall (default: 0.90)
        batch_size: Batch size for feature extraction (default: 64)
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
        backbone: Feature extractor to use ('inception' or 'resnet18')
        input_size: Input image size (default: 299 for inception, 224 for resnet18)
        rule: Filtering rule ('precision_and_auth', 'precision_only', 'auth_only')

    Returns
    -------
    {
      "precision_mask":   np.bool_(N_fake),
      "authenticity_mask":np.bool_(N_fake),
      "keep_mask":        np.bool_(N_fake),
      "indices":          np.ndarray[int],     # kept indices
      "metrics_before":   dict,
      "metrics_after":    dict,
      "filtered_tensor":  torch.Tensor | None  # if fake_source is a tensor
    }
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if input_size is None:
        input_size = 299 if backbone == "inception" else 224

    # --------------------------------------------------- embed once
    net, _ = _load_backbone(backbone, device)
    real_f = _embed(real_source, net, batch_size, device, input_size)
    fake_f = _embed(fake_source, net, batch_size, device, input_size)

    # Handle edge case: too few real samples for metrics
    if len(real_f) < 2:
        # With only 1 real sample, keep all synthetic samples (no meaningful filtering)
        precision_mask = np.ones(len(fake_f), dtype=bool)
        authenticity_mask = np.ones(len(fake_f), dtype=bool)
    else:
        # --------------------------------------------------- precision mask
        c_r = real_f.mean(0, keepdims=True)
        r_thr = np.quantile(np.linalg.norm(real_f - c_r, axis=1), alpha)
        precision_mask = np.linalg.norm(fake_f - c_r, axis=1) <= r_thr

        # --------------------------------------------------- authenticity mask
        nn_r2 = NearestNeighbors(n_neighbors=2).fit(real_f)
        d_r2 = nn_r2.kneighbors(real_f, return_distance=True)[0][:, 1]

        nn_fg = NearestNeighbors(n_neighbors=1).fit(real_f)
        d_fg, idx_fg = nn_fg.kneighbors(fake_f, return_distance=True)
        authenticity_mask = ~(d_fg.flatten() <= 2*d_r2[idx_fg.flatten()])

    # --------------------------------------------------- choose keep rule
    if rule == "precision_and_auth":
        keep_mask = precision_mask & authenticity_mask
    elif rule == "precision_only":
        keep_mask = precision_mask
    elif rule == "auth_only":
        keep_mask = authenticity_mask
    else:
        raise ValueError(rule)

    keep_idx = np.nonzero(keep_mask)[0]

    # --------------------------------------------------- metrics before / after
    metrics_before = dict(alpha_precision=float(precision_mask.mean()),
                          beta_recall=float(np.nan),          # recall unaffected by pruning
                          authenticity=float(authenticity_mask.mean()))

    # recompute full metrics for surviving samples
    # Package surviving samples into a Dataset/Tensor
    if isinstance(fake_source, torch.Tensor):
        filtered_tensor = fake_source[torch.from_numpy(keep_mask)]
        fake_ds_after = TensorDataset(filtered_tensor)
    else:
        filtered_tensor = None
        fake_ds_after = SubsetDataset(fake_source, keep_idx)

    metrics_after = evaluate_fidelity_diversity(
        real_source, fake_ds_after,
        alpha=alpha, beta=beta,
        batch_size=batch_size,
        device=device, backbone=backbone, input_size=input_size
    )

    return dict(precision_mask=precision_mask,
                authenticity_mask=authenticity_mask,
                keep_mask=keep_mask,
                indices=keep_idx,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                filtered_tensor=filtered_tensor)


# Helper: subset view for arbitrary Dataset -----------------------------------
class SubsetDataset(Dataset):
    def __init__(self, base: Dataset, indices: np.ndarray):
        self.base, self.idxs = base, indices
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        return self.base[self.idxs[i]]

# Helper: tensor wrapper if user wants to pass tensors directly ---------------
class TensorDataset(Dataset):
    def __init__(self, tensor: torch.Tensor):
        self.t = tensor
    def __len__(self): return self.t.shape[0]
    def __getitem__(self, idx):
        # dummy label 0
        return self.t[idx], 0
