from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from flwr_datasets.partitioner import DirichletPartitioner, PathologicalPartitioner, IidPartitioner
from datasets import Dataset as HFDataset


class BrainTumorDataset(Dataset):
    """
    This dataset loads brain MRI images and applies the following transformations:
    1. Convert to RGB (ensuring consistency across grayscale and color images)
    2. Resize to 64×64 pixels
    3. Convert to tensor (automatically normalizes [0, 255] → [0, 1])

    Labels:
    - 0 (Class 0): No tumor ("no" folder)
    - 1 (Class 1): Tumor present ("yes" folder)
    """

    def __init__(self, root: str = "data/brain_tumor_dataset"):
        """
        Initialize the Brain Tumor dataset.

        Args:
            root: Root directory containing "yes" and "no" subdirectories
        """
        self.root = Path(root)

        self.transform = transforms.Compose([
            # Step 1: Convert to RGB (handles both grayscale and color images)
            transforms.Lambda(lambda img: img.convert("RGB")),
            # Step 2: Resize to 64×64 pixels
            transforms.Resize((64, 64)),
            # Step 3: Convert to tensor (automatically applies x'_i = x_i / 255)
            transforms.ToTensor(),
        ])

        # Load using ImageFolder (expects "yes" and "no" subdirectories)
        # ImageFolder assigns labels alphabetically: "no"=0, "yes"=1
        self._dataset = datasets.ImageFolder(root=str(self.root), transform=self.transform)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """ Get a single sample from the dataset."""
        return self._dataset[idx]

    @property
    def targets(self):
        """Return list of all labels (for compatibility with partitioning functions)."""
        return self._dataset.targets

    @property
    def samples(self):
        """Return list of all (path, label) tuples."""
        return self._dataset.samples

    @property
    def classes(self):
        """Return list of class names."""
        return self._dataset.classes

    @property
    def class_to_idx(self):
        """Return mapping from class name to index."""
        return self._dataset.class_to_idx


def load_brain_tumor_data(
    partition_id: int,
    num_clients: int,
    non_iid_alpha: float,
    batch_size: int,
    total_n: int,
    partitioning: str,
    root: str,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load and partition the Brain Tumor dataset for federated learning.

    This function implements the data loading pipeline with:
    - Test set: 16.6% of total data
    - Training set subsampled to total_n samples (stratified)
    - Non-IID partitioning across clients using Flower Datasets partitioners

    Args:
        partition_id: Client ID (-1 for global test set)
        num_clients: Total number of clients
        non_iid_alpha: Dirichlet alpha for non-IID partitioning
        batch_size: Batch size for DataLoader
        total_n: Total number of training samples to use (before partitioning)
        partitioning: Partitioning strategy ("dirichlet", "extreme", "iid")
        root: Root directory of the dataset

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Load the full dataset
    full_dataset = BrainTumorDataset(root=root)

    # Step 1: Split into train/test with 16.6% test size
    train_indices, test_indices = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.166,
        stratify=full_dataset.targets,
        random_state=42,
        shuffle=True
    )

    # Step 2: Subsample training set to total_n samples (stratified)
    train_targets_full = [full_dataset.targets[i] for i in train_indices]
    train_indices_subsampled, _ = train_test_split(
        train_indices,
        train_size=min(total_n, len(train_indices)),
        stratify=train_targets_full,
        random_state=42
    )

    test_set = Subset(full_dataset, test_indices)

    # If requesting global test set, return it
    if partition_id == -1:
        return (
            DataLoader(test_set, batch_size=batch_size),
            DataLoader(test_set, batch_size=batch_size)
        )

    # Step 3: Partition the training data across clients using Flower Datasets
    partition_targets = [full_dataset.targets[i] for i in train_indices_subsampled]
    hf_train = HFDataset.from_dict({
        "label": partition_targets,
        "orig_idx": train_indices_subsampled
    })

    if partitioning == "extreme":
        partitioner = PathologicalPartitioner(
            num_partitions=num_clients,
            partition_by="label",
            num_classes_per_partition=1
        )
    elif non_iid_alpha == float('inf'):
        partitioner = IidPartitioner(num_partitions=num_clients)
    else:
        partitioner = DirichletPartitioner(
            num_partitions=num_clients,
            alpha=non_iid_alpha,
            partition_by="label"
        )
    
    partitioner.dataset = hf_train
    client_indices = partitioner.load_partition(partition_id)["orig_idx"]

    # Create client's dataset
    client_dataset = Subset(full_dataset, client_indices)

    # Create DataLoaders
    client_train_loader = DataLoader(
        client_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    client_test_loader = DataLoader(
        test_set,
        batch_size=batch_size
    )

    return client_train_loader, client_test_loader


def get_dataset_info(root: str = "../../data/brain_tumor_dataset") -> dict:
    """
    Get information about the Brain Tumor dataset.

    Args:
        root: Root directory of the dataset

    Returns:
        Dictionary with dataset statistics
    """
    dataset = BrainTumorDataset(root=root)
    targets = dataset.targets

    unique, counts = np.unique(targets, return_counts=True)
    class_distribution = dict(zip(unique, counts))

    return {
        "total_samples": len(dataset),
        "num_classes": len(dataset.classes),
        "classes": dataset.classes,
        "class_to_idx": dataset.class_to_idx,
        "class_distribution": {
            dataset.classes[idx]: count
            for idx, count in class_distribution.items()
        },
        "image_shape": (3, 64, 64),
        "value_range": "[0, 1] (normalized)",
    }

