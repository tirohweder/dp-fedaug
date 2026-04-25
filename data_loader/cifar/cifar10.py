import warnings
import logging

# Suppress warnings before importing datasets
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, PathologicalPartitioner, IidPartitioner
from typing import Tuple
from collections import Counter

_fds_cache: dict = {}


def _get_cached_fds(
    num_clients: int,
    non_iid_alpha: float,
    partitioning: str,
    cache_dir: str,
    seed: int
) -> FederatedDataset:
    """Get or create a cached FederatedDataset to avoid repeated initialization."""
    if isinstance(non_iid_alpha, str) and non_iid_alpha.lower() == "inf":
        non_iid_alpha = float("inf")

    cache_key = f"{num_clients}_{non_iid_alpha}_{partitioning}_{seed}"

    if cache_key not in _fds_cache:
        if partitioning == "extreme":
            partitioner = PathologicalPartitioner(
                num_partitions=num_clients,
                partition_by="label",
                num_classes_per_partition=1,
                seed=seed
            )
        elif non_iid_alpha == float("inf"):
            partitioner = IidPartitioner(num_partitions=num_clients)
        else:
            partitioner = DirichletPartitioner(
                num_partitions=num_clients,
                alpha=non_iid_alpha,
                partition_by="label",
                seed=seed
            )

        _fds_cache[cache_key] = FederatedDataset(
            dataset="cifar10",
            partitioners={"train": partitioner},
            cache_dir=cache_dir
        )

    return _fds_cache[cache_key]


def log_class_distribution(fds: FederatedDataset, num_clients: int, total_n: int) -> None:
    """Log the class distribution per client (only once at start)."""
    print("\n" + "=" * 60)
    print("📊 CLASS DISTRIBUTION PER CLIENT")
    print("=" * 60)

    train_len = len(fds.load_split("train"))
    scale = total_n / train_len if total_n < train_len else 1.0

    for client_id in range(num_clients):
        partition = fds.load_partition(client_id, "train")
        labels = [item["label"] for item in partition]

        n_samples = int(len(labels) * scale)
        labels = labels[:n_samples] if n_samples < len(labels) else labels

        label_counts = Counter(labels)
        dist_str = " | ".join([f"{k}:{v}" for k, v in sorted(label_counts.items())])
        print(f"  Client {client_id:2d} (n={len(labels):4d}): {dist_str}")

    print("=" * 60 + "\n")


def load_cifar10_federated_data(
    partition_id: int,
    num_clients: int,
    non_iid_alpha: float,
    batch_size: int,
    total_n: int,
    partitioning: str,
    cache_dir: str,
    seed: int = 42,
    log_distribution: bool = False,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Load and partition CIFAR-10 data for federated learning using Flower Datasets.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    fds = _get_cached_fds(num_clients, non_iid_alpha, partitioning, cache_dir, seed)

    if log_distribution and partition_id == -1:
        log_class_distribution(fds, num_clients, total_n)

    def apply_transforms(batch):
        # HF CIFAR-10 uses "img" while some datasets use "image"
        if "image" in batch:
            images = batch["image"]
        elif "img" in batch:
            images = batch["img"]
        else:
            image_key = None
            for key, values in batch.items():
                if isinstance(values, list) and values:
                    if isinstance(values[0], Image.Image):
                        image_key = key
                        break
            if image_key is None:
                raise KeyError(f"Expected image key in batch; found keys: {list(batch.keys())}")
            images = batch[image_key]

        batch["image"] = [transform(img) for img in images]
        if "img" in batch:
            del batch["img"]
        return batch

    test_set = fds.load_split("test").with_transform(apply_transforms)

    def collate_fn(batch):
        images = torch.stack([item["image"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch])
        return images, labels

    if partition_id == -1:
        full_dataset = fds.load_split("train").with_transform(apply_transforms)
        if total_n < len(full_dataset):
            full_dataset = full_dataset.select(range(total_n))
        return DataLoader(full_dataset, batch_size=batch_size, collate_fn=collate_fn), \
               DataLoader(test_set, batch_size=batch_size, collate_fn=collate_fn)

    client_dataset = fds.load_partition(partition_id, "train").with_transform(apply_transforms)

    train_len = len(fds.load_split("train"))
    if total_n < train_len:
        scale = total_n / train_len
        n_samples = int(len(client_dataset) * scale)
        if n_samples > 0:
            client_dataset = client_dataset.select(range(min(n_samples, len(client_dataset))))

    return DataLoader(client_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn), \
           DataLoader(test_set, batch_size=batch_size, collate_fn=collate_fn)
