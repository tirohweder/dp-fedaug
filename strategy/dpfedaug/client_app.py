import warnings
import logging
# Suppress warnings before any other imports
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import torch
import numpy as np
from flwr.app import ArrayRecord, Context, Message, RecordDict, MetricRecord, ConfigRecord
from flwr.clientapp import ClientApp
from opacus.accountants.utils import get_noise_multiplier
from strategy.dpfedaug.task import (
    get_model,
    load_data,
    get_dpfedaug_generator,
    train_fn,
    test_fn,
    CustomTensorDataset,
    get_common_config
)
from torch.utils.data import ConcatDataset, DataLoader

app = ClientApp()

def _parse_optional_float(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == "none":
        return None
    return float(value)


class DictTensorDataset(torch.utils.data.Dataset):
    """Dataset that returns dict samples compatible with HF-style collate."""
    def __init__(self, samples: torch.Tensor, labels: torch.Tensor):
        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        return {"image": self.samples[index], "label": int(self.labels[index])}

    def __len__(self):
        return self.samples.size(0)

@app.query("fedaug_generate")
def fedaug_generate(msg: Message, context: Context) -> Message:
    """Generate synthetic samples using DP-VAE."""
    cfg = context.run_config
    common = get_common_config(cfg)
    
    # Extract specific params
    synthetic_count = cfg["synthetic-count"]
    balancing = cfg["balancing"]
    
    # Extract DP parameters
    target_epsilon_raw = cfg["target-epsilon"]
    delta = cfg["synthetic-delta"]
    syn_epochs = cfg["synthetic-epochs"]
    syn_batch_size = cfg["synthetic-batch-size"]
    
    # Load data first to know the dataset size for noise computation
    trainloader, _ = load_data(
        partition_id=context.node_config["partition-id"],
        **common
    )
    
    # Get dataset size for sample_rate calculation
    dataset_size = len(trainloader.dataset)
    
    # Process target_epsilon and compute noise_multiplier
    # target_epsilon can be: "none" or "0" (no DP), or a numeric string like "1" or "8"
    # Parse the raw value first
    if isinstance(target_epsilon_raw, str):
        target_epsilon_raw_lower = target_epsilon_raw.lower()
        if target_epsilon_raw_lower == "none":
            target_epsilon = None
        else:
            target_epsilon = float(target_epsilon_raw)
    else:
        target_epsilon = float(target_epsilon_raw)
    
    # Determine if DP should be applied
    # ε=0 means "perfect privacy" which is impossible, treat as no DP
    # ε=None also means no DP
    if target_epsilon is None or target_epsilon == 0:
        # No DP - use noise_multiplier = 0
        noise_multiplier = 0.0
        print(f"[DP-FedAug] No DP enabled (ε = ∞)")
    else:
        # Minimum epsilon that Opacus can handle (very strong privacy)
        MIN_EPSILON = 0.1
        if target_epsilon < MIN_EPSILON:
            print(f"[DP-FedAug] Warning: ε={target_epsilon} is too small, using minimum ε={MIN_EPSILON}")
            target_epsilon = MIN_EPSILON
        
        # Compute sample_rate for the privacy accountant
        # Opacus requires sample_rate < 1, so we cap it
        # If batch_size >= dataset_size, we use the full dataset per step
        effective_batch_size = min(syn_batch_size, dataset_size)
        sample_rate = effective_batch_size / dataset_size
        
        # Ensure sample_rate is strictly less than 1 (Opacus requirement)
        # This can happen when batch_size == dataset_size
        if sample_rate >= 1.0:
            sample_rate = 0.99
            print(f"[DP-FedAug] Warning: batch_size >= dataset_size, capping sample_rate to {sample_rate}")
        
        # Calculate noise_multiplier to achieve target_epsilon
        noise_multiplier = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=syn_epochs,
        )
        print(f"[DP-FedAug] Target ε={target_epsilon}, δ={delta}, "
              f"sample_rate={sample_rate:.4f}, epochs={syn_epochs} → noise_multiplier={noise_multiplier:.4f}")

    # Synthetic specific params (explicitly loaded, no defaults)
    syn_params = {
        "epochs": syn_epochs,
        "batch_size": syn_batch_size,
        "latent_dim": cfg["synthetic-latent-dim"],
        "kl_warmup": cfg["synthetic-kl-warmup"],
        "lr": cfg["synthetic-lr"],
        "delta": delta,
        "max_grad_norm": cfg["max-grad-norm"],
        "img_size": cfg["img-size"],
        "synthetic_count": synthetic_count,
        "scale_syn": (balancing == "scaled"),
        "seed": common["seed"],
        "noise_multiplier": noise_multiplier,
        "eval_metrics": cfg.get("synthetic-eval-metrics", True),
    }

    if len(trainloader.dataset) == 0:
        content = RecordDict({
            "status": ConfigRecord({
                "success": False,
                "reason": "empty_dataset",
                "partition_id": context.node_config["partition-id"]
            })
        })
        return Message(content=content, reply_to=msg)

    client_samples, client_labels = [], []
    for samples, labels in trainloader:
        client_samples.append(samples)
        client_labels.append(labels)
    if not client_samples or not client_labels:
        content = RecordDict({
            "status": ConfigRecord({
                "success": False,
                "reason": "empty_dataset",
                "partition_id": context.node_config["partition-id"]
            })
        })
        return Message(content=content, reply_to=msg)
    client_x = torch.cat(client_samples, dim=0)
    client_y = torch.cat(client_labels, dim=0)

    # Get the DP generator factory
    train_and_generate = get_dpfedaug_generator(common["dataset_name"])
    
    # Run DP generative training
    synthetic_x, synthetic_y, metrics, epsilon_per_label = train_and_generate(
        data_tensor=client_x,
        label_tensor=client_y,
        **syn_params
    )

    eps_values = list(epsilon_per_label.values())
    content = RecordDict({
        "samples": ArrayRecord([synthetic_x.numpy()]),
        "labels": ArrayRecord([synthetic_y.numpy()]),
        "epsilon_mean": ArrayRecord([np.array([np.mean(eps_values)])]),
        "epsilon_max": ArrayRecord([np.array([np.max(eps_values)])]),
    })
    return Message(content=content, reply_to=msg)

@app.query("fedaug_store_data")
def fedaug_store_data(msg: Message, context: Context) -> Message:
    """Receives shuffled synthetic data from server and saves to state."""
    context.state["synthetic_samples"] = msg.content["samples"]
    context.state["synthetic_labels"] = msg.content["labels"]
    return Message(content=RecordDict({"status": ConfigRecord({"success": True})}), reply_to=msg)

@app.train()
def train(msg: Message, context: Context):
    """Train classification model on augmented local data."""
    cfg = context.run_config
    common = get_common_config(cfg)

    num_local_epochs = cfg["num-local-epochs"]
    gradient_clipping = cfg["gradient_clipping"]
    weight_decay = cfg["weight-decay"]

    # --- Parse DP-SGD config ---
    dp_enabled_raw = cfg.get("updates-dp-enabled", False)
    if isinstance(dp_enabled_raw, str):
        dp_enabled = dp_enabled_raw.lower() == "true"
    else:
        dp_enabled = bool(dp_enabled_raw)

    dp_epsilon = _parse_optional_float(cfg.get("updates-dp-epsilon")) if dp_enabled else None
    dp_delta = _parse_optional_float(cfg.get("updates-dp-delta")) if dp_enabled else None
    if dp_enabled:
        # Accept both the current and legacy config names while experiments are being aligned.
        dp_max_grad_norm_raw = cfg.get("updates-dp-max-grad-norm")
        if dp_max_grad_norm_raw is None:
            dp_max_grad_norm_raw = cfg.get("updates-dp-clipping-norm", 1.0)
        dp_max_grad_norm = float(dp_max_grad_norm_raw)
    else:
        dp_max_grad_norm = 1.0
    num_server_rounds = int(cfg.get("num-server-rounds", 50))
    
    model = get_model(
        common["dataset_name"],
        mnist_use_dropout=common["mnist_use_dropout"],
        mnist_dropout_rate=common["mnist_dropout_rate"],
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainloader_original, _ = load_data(
        partition_id=context.node_config["partition-id"],
        **common
    )
    real_dataset_size = len(trainloader_original.dataset)

    if "synthetic_samples" in context.state:
        samples_np = context.state["synthetic_samples"].to_numpy_ndarrays()[0]
        labels_np = context.state["synthetic_labels"].to_numpy_ndarrays()[0]
        samples_tensor = torch.from_numpy(samples_np)
        labels_tensor = torch.from_numpy(labels_np)

        try:
            first_item = trainloader_original.dataset[0]
            use_dict_samples = isinstance(first_item, dict)
        except Exception:
            use_dict_samples = False

        if use_dict_samples:
            synth_ds = DictTensorDataset(samples_tensor, labels_tensor)
            trainloader = DataLoader(
                ConcatDataset([trainloader_original.dataset, synth_ds]),
                batch_size=common["batch_size"],
                shuffle=True,
                collate_fn=trainloader_original.collate_fn,
            )
        else:
            synth_ds = CustomTensorDataset(samples_tensor, labels_tensor)
            augmented_ds = ConcatDataset([trainloader_original.dataset, synth_ds])
            trainloader = DataLoader(augmented_ds, batch_size=common["batch_size"], shuffle=True)
    else:
        trainloader = trainloader_original

    train_loss, epsilon_spent = train_fn(
        net=model,
        trainloader=trainloader,
        epochs=num_local_epochs,
        lr=msg.content["config"]["lr"],
        device=device,
        gradient_clipping=gradient_clipping,
        dataset_name=common["dataset_name"],
        weight_decay=weight_decay,
        classification_type=common["classification_type"],
        dp_enabled=dp_enabled,
        dp_epsilon=dp_epsilon,
        dp_delta=dp_delta,
        dp_max_grad_norm=dp_max_grad_norm,
        num_server_rounds=num_server_rounds,
        real_dataset_size=real_dataset_size,
    )

    metrics = {"train_loss": float(train_loss), "num-examples": int(len(trainloader.dataset))}
    if epsilon_spent is not None:
        metrics["updates_dp_epsilon_spent"] = float(epsilon_spent)

    return Message(content=RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord(metrics)
    }), reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate model on the global test set or local data."""
    cfg = context.run_config
    common = get_common_config(cfg)

    model = get_model(
        common["dataset_name"],
        mnist_use_dropout=common["mnist_use_dropout"],
        mnist_dropout_rate=common["mnist_dropout_rate"],
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Check if we should evaluate on local data (for per-client fairness metrics)
    eval_local = msg.content.get("config", ConfigRecord()).get("eval_local", False)

    if eval_local:
        # Evaluate global model on this client's local training partition
        trainloader, _ = load_data(
            partition_id=context.node_config["partition-id"],
            **common
        )
        metrics_tuple = test_fn(model, trainloader, device, common["dataset_name"], common["classification_type"])
        metrics_keys = ["loss", "accuracy", "auc", "precision", "recall", "f1", "ap"]
        metrics_dict = dict(zip(metrics_keys, metrics_tuple))
        metrics_dict["num-examples"] = len(trainloader.dataset)
        metrics_dict["partition_id"] = float(context.node_config["partition-id"])
    else:
        _, testloader = load_data(
            partition_id=context.node_config["partition-id"],
            **common
        )
        metrics_tuple = test_fn(model, testloader, device, common["dataset_name"], common["classification_type"])
        metrics_keys = ["loss", "accuracy", "auc", "precision", "recall", "f1", "ap"]
        metrics_dict = dict(zip(metrics_keys, metrics_tuple))
        metrics_dict["num-examples"] = len(testloader.dataset)

    return Message(content=RecordDict({"metrics": MetricRecord(metrics_dict)}), reply_to=msg)
