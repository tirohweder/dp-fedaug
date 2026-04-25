"""
Flower ServerApp for FedProx.
"""
import time
from logging import INFO
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord
from flwr.common.logger import log
from flwr.serverapp import Grid, ServerApp

from strategy.fedprox.strategy import FedProxStrategy
from strategy.fedprox.task import get_model, load_data, test_fn

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    log(INFO, "ServerApp main started (FedProx).")

    # 1. Get configuration from pyproject.toml
    num_rounds = context.run_config["num-server-rounds"]
    dataset_name = context.run_config["dataset"]
    num_clients = context.run_config["num-clients"]
    non_iid_alpha = context.run_config.get("non-iid-alpha", 1.0)  # Default for extreme partitioning
    base_lr = context.run_config["lr"]

    # Additional configs
    total_n = context.run_config.get("total-n", 200)
    weight_decay = context.run_config.get("weight-decay", 1e-4)
    seed = context.run_config.get("seed", 42)
    proximal_mu = context.run_config.get("proximal-mu", 0.0)
    wandb_project = context.run_config.get("wandb-project", "FedProx-Baseline-Braintumor")
    classification_type = context.run_config.get("classification_type", "binary")
    partitioning = context.run_config.get("partitioning", "dirichlet")

    # Set seed on server
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 2. Define the centralized evaluation function (evaluate_fn)
    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> Optional[MetricRecord]:
        """Evaluate the global model on the centralized (global) test set."""
        model = get_model(dataset_name)
        model.load_state_dict(arrays.to_torch_state_dict(), strict=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load the global test set (using partition_id = -1)
        _, testloader = load_data(
            dataset_name=dataset_name,
            partition_id=-1,
            num_clients=num_clients,
            non_iid_alpha=non_iid_alpha,
            batch_size=64,
            total_n=total_n,
            partitioning=partitioning,
            seed=seed,
        )

        loss, accuracy, auc, precision, recall, f1, ap = test_fn(
            model,
            testloader,
            device,
            dataset_name,
            classification_type,
        )

        log(
            INFO,
            f"Global Test Set: Round {server_round}, "
            f"Loss: {loss:.4f}, Acc: {accuracy:.4f}, AUC: {auc:.4f}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AP: {ap:.4f}",
        )

        return MetricRecord(
            {
                "accuracy": accuracy,
                "loss": loss,
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "ap": ap,
            }
        )

    # 3. Create a unique run directory for artifacts
    run_dir_path = Path(f"results/run_{time.strftime('%Y%m%d_%H%M%S')}_fedprox")
    run_dir_path.mkdir(parents=True, exist_ok=True)
    run_name = run_dir_path.name

    # 4. Initialize the Strategy
    strategy = FedProxStrategy(
        fraction_train=1.0,  # Train on all clients
        fraction_evaluate=1.0,  # Evaluate on all clients
        min_available_nodes=int(context.run_config["num-clients"]),
    )
    strategy.set_save_path_and_run_dir(run_dir_path, run_name)
    strategy.set_wandb_project(wandb_project)
    strategy.set_run_config(
        {
            "dataset": dataset_name,
            "num_clients": num_clients,
            "non_iid_alpha": non_iid_alpha,
            "partitioning": partitioning,
            "total_n": total_n,
            "lr": base_lr,
            "num_local_epochs": context.run_config["num-local-epochs"],
            "batch_size": context.run_config["batch-size"],
            "gradient_clipping": context.run_config["gradient_clipping"],
            "strategy": "FedProx",
            "weight_decay": weight_decay,
            "seed": seed,
            "proximal_mu": proximal_mu,
        }
    )

    # 5. Load initial model
    initial_model = get_model(dataset_name)
    initial_arrays = ArrayRecord(initial_model.state_dict())

    # 6. Define the training config to be sent to clients
    train_config = ConfigRecord(
        {
            "lr": base_lr,
            "mu": proximal_mu,
            "seed": seed,
        }
    )

    # 7. Start the federated learning run
    log(INFO, f"Starting run: {run_name} with {num_clients} clients (FedProx)...")
    strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        train_config=train_config,
        evaluate_fn=evaluate_fn,
    )
    log(INFO, f"Run {run_name} finished.")


