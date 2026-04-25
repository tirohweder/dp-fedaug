import warnings
import logging

# Suppress all warnings at module load time (before any imports that generate warnings)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress verbose logging from datasets/huggingface
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("datasets.builder").setLevel(logging.ERROR)
logging.getLogger("datasets.info").setLevel(logging.ERROR)

from typing import Tuple, Callable, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
import numpy as np

# ---
# Constants
# ---
DEFAULT_DATA_PATHS = {
    "braintumor": "data/brain_tumor_dataset",
    "mnist": "data/mnist",
    "cifar10": "data/cifar10"
}

# ---
# 0. Custom Dataset for VAE and Augmentation
# ---
class CustomTensorDataset(Dataset):
    """Custom Dataset wrapper for tensors."""
    def __init__(self, *tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        # Return image tensor and label as int (to match ImageFolder format)
        if len(self.tensors) == 1:
            return self.tensors[0][index]
        return self.tensors[0][index], int(self.tensors[1][index])

    def __len__(self):
        return self.tensors[0].size(0)

def get_model(
    dataset_name: str,
    mnist_use_dropout: bool = False,
    mnist_dropout_rate: float = 0.1,
) -> nn.Module:
    """Factory for classification models."""
    if dataset_name == "braintumor":
        from models.braintumor.brain_tumor_cnn import BrainTumorCNN
        return BrainTumorCNN()
    elif dataset_name == "mnist":
        from models.mnist.mnist_cnn import MNIST_CNN
        return MNIST_CNN(
            use_dropout=mnist_use_dropout,
            dropout_rate=mnist_dropout_rate,
        )
    elif dataset_name == "cifar10":
        from models.cifar.cifar_cnn import CIFAR_CNN
        return CIFAR_CNN()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_dpfedaug_generator(dataset_name: str) -> Callable:
    """Factory for DP generative training and generation functions."""
    if dataset_name == "braintumor":
        from models.braintumor.train_braintumor_vae import train_braintumor_vae_dp
        return train_braintumor_vae_dp
    elif dataset_name == "mnist":
        from models.mnist.train_mnist_vae import train_mnist_vae_dp
        return train_mnist_vae_dp
    elif dataset_name == "cifar10":
        from models.cifar.train_cifar_vae import train_cifar_vae_dp
        return train_cifar_vae_dp
    else:
        raise ValueError(f"DP-FedAug not supported for dataset: {dataset_name}")

def get_common_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts common configuration parameters to reduce duplication."""
    mnist_use_dropout = cfg.get("mnist-use-dropout", False)
    if isinstance(mnist_use_dropout, str):
        mnist_use_dropout = mnist_use_dropout.lower() == "true"

    return {
        "dataset_name": cfg["dataset"],
        "num_clients": cfg["num-clients"],
        "non_iid_alpha": cfg["non-iid-alpha"],
        "batch_size": cfg["batch-size"],
        "total_n": cfg["total-n"],
        "partitioning": cfg["partitioning"],
        "seed": cfg["seed"],
        "classification_type": cfg["classification_type"],
        "mnist_use_dropout": mnist_use_dropout,
        "mnist_dropout_rate": float(cfg.get("mnist-dropout-rate", 0.1)),
    }

# ---
# 2. Training & Evaluation Functions
# ---

def train_fn(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    gradient_clipping: bool,
    dataset_name: str,
    weight_decay: float,
    classification_type: str,
    dp_enabled: bool = False,
    dp_epsilon: float = None,
    dp_delta: float = None,
    dp_max_grad_norm: float = 1.0,
    num_server_rounds: int = None,
    real_dataset_size: int = None,
) -> Tuple[float, float]:
    """Train the model on the client's data_loader.

    When dp_enabled=True, wraps training with Opacus DP-SGD.
    Privacy is accounted for the REAL local data only (real_dataset_size),
    even when the trainloader contains additional synthetic samples.

    Returns:
        (train_loss, epsilon_spent) — epsilon_spent is None when dp_enabled=False.
    """
    if classification_type == "binary":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    net.to(device)

    privacy_engine = None
    epsilon_spent = None

    if dp_enabled:
        from opacus import PrivacyEngine
        from opacus.accountants.utils import get_noise_multiplier

        # Privacy accounting is over the REAL local data, not synthetic.
        # Synthetic data is already DP-protected by the VAE generation step.
        n_real = real_dataset_size if real_dataset_size is not None else len(trainloader.dataset)
        batch_size = trainloader.batch_size or 32
        sample_rate = min(batch_size / n_real, 0.99)

        # Total gradient steps across the ENTIRE FL training (all rounds).
        # This ensures the stated ε covers full training, not just one round.
        steps_per_round = max(1, n_real // batch_size)
        total_steps = (num_server_rounds or 1) * epochs * steps_per_round

        noise_multiplier = get_noise_multiplier(
            target_epsilon=dp_epsilon,
            target_delta=dp_delta,
            sample_rate=sample_rate,
            steps=total_steps,
        )
        print(
            f"[DP-SGD] ε={dp_epsilon}, δ={dp_delta}, "
            f"noise_multiplier={noise_multiplier:.4f}, "
            f"sample_rate={sample_rate:.4f}, total_steps={total_steps}"
        )

        privacy_engine = PrivacyEngine()
        net, optimizer, trainloader = privacy_engine.make_private(
            module=net,
            optimizer=optimizer,
            data_loader=trainloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=dp_max_grad_norm,
        )
        # Opacus clips per-sample gradients internally — skip manual clipping.
        gradient_clipping = False

    net.train()
    total_loss = 0.0

    for epoch in range(epochs):
        for images, labels in trainloader:
            if len(labels) == 0:
                continue
            images, labels = images.to(device), labels.to(device)

            if classification_type == "binary":
                labels = labels.unsqueeze(1).float()
            else:
                labels = labels.long().view(-1)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()

            if gradient_clipping:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

    if dp_enabled and privacy_engine is not None:
        epsilon_spent = privacy_engine.get_epsilon(delta=dp_delta)
        print(f"[DP-SGD] Cumulative ε spent this round: {epsilon_spent:.4f}")
        net = net._module  # unwrap GradSampleModule → original model

    if len(trainloader) == 0 or epochs == 0:
        return 0.0, epsilon_spent

    return total_loss / (len(trainloader) * epochs), epsilon_spent


def test_fn(
    net: nn.Module,
    testloader: DataLoader,
    device: torch.device,
    dataset_name: str,
    classification_type: str,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Test the model on the global test set.

    Returns:
        (loss, accuracy, auc, precision, recall, f1, ap)
    """

    if classification_type == "binary":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    net.to(device)
    net.eval()

    loss, correct, total = 0.0, 0, 0
    all_labels = []
    all_preds_auc = []
    all_preds_class = []  # For precision/recall/f1

    with torch.no_grad():
        for images, labels in testloader:
            if len(labels) == 0: continue # Skip empty batches
            images, labels = images.to(device), labels.to(device)

            if classification_type == "binary":
                labels_formatted = labels.unsqueeze(1).float()
            else:
                # Ensure labels is 1D for CrossEntropyLoss (handles any extra dims)
                labels_formatted = labels.long().view(-1)

            outputs = net(images)

            # Handle empty output
            if outputs.size(0) == 0:
                continue

            loss += criterion(outputs, labels_formatted).item()

            # --- Calculate Accuracy and store predictions ---
            if classification_type == "binary":
                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds == labels_formatted).sum().item()
                # Store class predictions (0 or 1)
                all_preds_class.extend(preds.cpu().numpy().flatten())
            else:
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels_formatted).sum().item()
                all_preds_class.extend(preds.cpu().numpy())

            total += labels.size(0)

            # --- Store for AUC and AP ---
            all_labels.extend(labels.cpu().numpy())

            if classification_type == "binary":
                all_preds_auc.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
            else:
                all_preds_auc.extend(F.softmax(outputs, dim=1).cpu().numpy())

    if len(testloader) == 0 or total == 0:
        return 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5 # Return defaults if no data

    avg_loss = loss / len(testloader)
    accuracy = correct / total

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds_class = np.array(all_preds_class)
    all_preds_auc = np.array(all_preds_auc)

    # --- Calculate AUC ---
    try:
        if classification_type == "binary":
            auc = roc_auc_score(all_labels, all_preds_auc)
        else:
            # Check if all classes are present in labels, otherwise OVR breaks
            present_classes = np.unique(all_labels)
            if len(present_classes) < all_preds_auc.shape[1]:
                try:
                    auc = roc_auc_score(all_labels, all_preds_auc, multi_class='ovo')
                except ValueError:
                    auc = 0.5 # Fallback
            else:
                auc = roc_auc_score(all_labels, all_preds_auc, multi_class='ovr')
    except ValueError:
        auc = 0.5

    # --- Calculate Precision, Recall, F1-Score ---
    try:
        if classification_type == "binary":
            # Binary classification
            precision = precision_score(all_labels, all_preds_class, zero_division=0)
            recall = recall_score(all_labels, all_preds_class, zero_division=0)
            f1 = f1_score(all_labels, all_preds_class, zero_division=0)
        else:
            # Multi-class: use macro averaging
            precision = precision_score(all_labels, all_preds_class, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds_class, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds_class, average='macro', zero_division=0)
    except ValueError:
        precision, recall, f1 = 0.0, 0.0, 0.0

    # --- Calculate Average Precision (AP) ---
    try:
        if classification_type == "binary":
            ap = average_precision_score(all_labels, all_preds_auc)
        else:
            ap = average_precision_score(all_labels, all_preds_auc, average='macro')
    except (ValueError, IndexError):
        ap = 0.5

    return avg_loss, accuracy, auc, precision, recall, f1, ap

def load_data(
    dataset_name: str,
    partition_id: int,
    num_clients: int,
    non_iid_alpha: float,
    batch_size: int,
    total_n: int,
    partitioning: str,
    seed: int,
    dataset_path: str = None,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Factory for data loaders.
    If dataset_path is None, it uses default paths defined in DEFAULT_DATA_PATHS.
    """
    path = dataset_path if dataset_path else DEFAULT_DATA_PATHS.get(dataset_name)
    
    if dataset_name == "braintumor":
        from data_loader.braintumor.brain_tumor import load_brain_tumor_data
        return load_brain_tumor_data(
            partition_id=partition_id,
            num_clients=num_clients,
            non_iid_alpha=non_iid_alpha,
            batch_size=batch_size,
            total_n=total_n,
            partitioning=partitioning,
            root=path
        )
    elif dataset_name == "mnist":
        from data_loader.mnist.mnist import load_mnist_federated_data
        return load_mnist_federated_data(
            partition_id=partition_id,
            num_clients=num_clients,
            non_iid_alpha=non_iid_alpha,
            batch_size=batch_size,
            total_n=total_n,
            partitioning=partitioning,
            cache_dir=path,
            seed=seed,
            **kwargs
        )
    elif dataset_name == "cifar10":
        from data_loader.cifar.cifar10 import load_cifar10_federated_data
        return load_cifar10_federated_data(
            partition_id=partition_id,
            num_clients=num_clients,
            non_iid_alpha=non_iid_alpha,
            batch_size=batch_size,
            total_n=total_n,
            partitioning=partitioning,
            cache_dir=path,
            seed=seed,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
