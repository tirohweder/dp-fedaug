import warnings
import logging
# Suppress warnings before any other imports
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import time
from logging import INFO
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common.logger import log
from flwr.serverapp import Grid, ServerApp
from strategy.dpfedaug.strategy import DPFedAvg
from strategy.dpfedaug.task import get_model, load_data, test_fn

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    # 1. Centralized config loading
    cfg = context.run_config
    num_rounds = cfg["num-server-rounds"]
    dataset_name = cfg["dataset"]
    num_clients = cfg["num-clients"]
    non_iid_alpha = cfg["non-iid-alpha"]
    base_lr = cfg["lr"]
    seed = cfg["seed"]
    total_n = cfg["total-n"]
    partitioning = cfg["partitioning"]
    wandb_project = cfg["wandb-project"]
    classification_type = cfg.get("classification_type", "binary")
    mnist_use_dropout_raw = cfg.get("mnist-use-dropout", False)
    if isinstance(mnist_use_dropout_raw, str):
        mnist_use_dropout = mnist_use_dropout_raw.lower() == "true"
    else:
        mnist_use_dropout = bool(mnist_use_dropout_raw)
    mnist_dropout_rate = float(cfg.get("mnist-dropout-rate", 0.1))

    np.random.seed(seed); torch.manual_seed(seed)

    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> Optional[MetricRecord]:
        model = get_model(
            dataset_name,
            mnist_use_dropout=mnist_use_dropout,
            mnist_dropout_rate=mnist_dropout_rate,
        )
        model.load_state_dict(arrays.to_torch_state_dict())
        
        _, testloader = load_data(
            dataset_name=dataset_name,
            partition_id=-1,
            num_clients=num_clients,
            non_iid_alpha=non_iid_alpha,
            batch_size=64, # Evaluation batch size
            total_n=total_n,
            partitioning=partitioning,
            seed=seed,
            log_distribution=(server_round == 0)  # Log distribution only on first evaluation
        )
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        metrics = test_fn(model, testloader, device, dataset_name, classification_type)
        return MetricRecord(dict(zip(["loss", "accuracy", "auc", "precision", "recall", "f1", "ap"], metrics)))

    run_dir = Path(f"results/dp_run_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)

    strategy = DPFedAvg(fraction_train=1.0, fraction_evaluate=1.0, min_available_nodes=num_clients)
    strategy.set_save_path_and_run_dir(run_dir, run_dir.name)
    strategy.set_wandb_project(wandb_project)
    strategy.set_run_config(context.run_config)

    strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(
            get_model(
                dataset_name,
                mnist_use_dropout=mnist_use_dropout,
                mnist_dropout_rate=mnist_dropout_rate,
            ).state_dict()
        ),
        num_rounds=num_rounds,
        train_config=ConfigRecord({"lr": base_lr, "seed": seed}),
        evaluate_fn=evaluate_fn
    )
