"""FedProx task utilities."""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score

# Import models
from models import BrainTumorCNN, MNIST_CNN

# Import data loaders
from data_loader import load_brain_tumor_data, load_mnist_federated_data

# Default data paths
DEFAULT_DATA_PATHS = {
    "braintumor": "data/brain_tumor_dataset",
    "mnist": "data/mnist"
}


def get_model(dataset_name: str) -> nn.Module:
    """Returns the appropriate model based on the dataset name."""
    if dataset_name == "braintumor":
        return BrainTumorCNN()
    elif dataset_name == "mnist":
        return MNIST_CNN()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_data(
    dataset_name: str,
    partition_id: int,
    num_clients: int,
    non_iid_alpha: float,
    batch_size: int,
    total_n: int,
    partitioning: str = "dirichlet",
    seed: int = 42,
    dataset_path: str = None,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """Factory for data loaders."""
    path = dataset_path if dataset_path else DEFAULT_DATA_PATHS.get(dataset_name)

    if dataset_name == "braintumor":
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
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def test_fn(
    net: nn.Module,
    testloader: DataLoader,
    device: torch.device,
    dataset_name: str,
    classification_type: str,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Test the model on the test set.

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
    all_preds_class = []

    with torch.no_grad():
        for images, labels in testloader:
            if len(labels) == 0:
                continue
            images, labels = images.to(device), labels.to(device)

            if classification_type == "binary":
                labels_formatted = labels.unsqueeze(1).float()
            else:
                labels_formatted = labels.long().view(-1)

            outputs = net(images)

            if outputs.size(0) == 0:
                continue

            loss += criterion(outputs, labels_formatted).item()

            if classification_type == "binary":
                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds == labels_formatted).sum().item()
                all_preds_class.extend(preds.cpu().numpy().flatten())
            else:
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels_formatted).sum().item()
                all_preds_class.extend(preds.cpu().numpy())

            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())

            if classification_type == "binary":
                all_preds_auc.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
            else:
                all_preds_auc.extend(F.softmax(outputs, dim=1).cpu().numpy())

    if len(testloader) == 0 or total == 0:
        return 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5

    avg_loss = loss / len(testloader)
    accuracy = correct / total

    all_labels = np.array(all_labels)
    all_preds_class = np.array(all_preds_class)
    all_preds_auc = np.array(all_preds_auc)

    # AUC
    try:
        if classification_type == "binary":
            auc = roc_auc_score(all_labels, all_preds_auc)
        else:
            present_classes = np.unique(all_labels)
            if len(present_classes) < all_preds_auc.shape[1]:
                try:
                    auc = roc_auc_score(all_labels, all_preds_auc, multi_class='ovo')
                except ValueError:
                    auc = 0.5
            else:
                auc = roc_auc_score(all_labels, all_preds_auc, multi_class='ovr')
    except ValueError:
        auc = 0.5

    # Precision, Recall, F1
    try:
        if classification_type == "binary":
            precision = precision_score(all_labels, all_preds_class, zero_division=0)
            recall = recall_score(all_labels, all_preds_class, zero_division=0)
            f1 = f1_score(all_labels, all_preds_class, zero_division=0)
        else:
            precision = precision_score(all_labels, all_preds_class, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds_class, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds_class, average='macro', zero_division=0)
    except ValueError:
        precision, recall, f1 = 0.0, 0.0, 0.0

    # Average Precision
    try:
        if classification_type == "binary":
            ap = average_precision_score(all_labels, all_preds_auc)
        else:
            ap = average_precision_score(all_labels, all_preds_auc, average='macro')
    except ValueError:
        ap = 0.5

    return avg_loss, accuracy, auc, precision, recall, f1, ap


def train_fn_prox(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    gradient_clipping: bool,
    dataset_name: str,
    weight_decay: float,
    mu: float,
    global_params: dict,
    classification_type: str = "binary",
) -> float:
    """Train with FedProx proximal term."""
    if classification_type == "binary":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    net.to(device)
    net.train()

    # Move global params to device once
    global_params = {k: v.to(device) for k, v in global_params.items()}

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

            # Proximal term: (mu/2) * ||w - w0||^2
            if mu > 0.0:
                prox_term = 0.0
                for name, param in net.named_parameters():
                    if name in global_params:
                        prox_term = prox_term + torch.sum((param - global_params[name]) ** 2)
                loss = loss + (mu / 2.0) * prox_term

            loss.backward()

            if gradient_clipping:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

    if len(trainloader) == 0 or epochs == 0:
        return 0.0

    return total_loss / (len(trainloader) * epochs)
