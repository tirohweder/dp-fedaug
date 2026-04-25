"""
Flower ClientApp for FedProx.
"""

import numpy as np
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from strategy.fedprox.task import get_model, load_data, test_fn, train_fn_prox

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data with FedProx proximal term."""

    # Get config
    dataset_name = context.run_config["dataset"]
    num_clients = context.run_config["num-clients"]
    non_iid_alpha = context.run_config.get("non-iid-alpha", 1.0)  # Default for extreme partitioning
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["num-local-epochs"]
    gradient_clipping = context.run_config["gradient_clipping"]
    total_n = context.run_config.get("total-n", 200)
    weight_decay = context.run_config.get("weight-decay", 1e-4)
    classification_type = context.run_config.get("classification_type", "binary")
    partitioning = context.run_config.get("partitioning", "dirichlet")

    lr = msg.content["config"]["lr"]
    mu = msg.content["config"].get("mu", 0.0)
    partition_id = context.node_config["partition-id"]

    # Set seed for local training
    seed = msg.content["config"].get("seed", 42)
    np.random.seed(seed + partition_id)
    torch.manual_seed(seed + partition_id)

    model = get_model(dataset_name)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Snapshot global parameters for proximal term
    global_params = {k: v.clone().detach() for k, v in model.state_dict().items()}

    # Load the local dataset
    trainloader, _ = load_data(
        dataset_name=dataset_name,
        partition_id=partition_id,
        num_clients=num_clients,
        non_iid_alpha=non_iid_alpha,
        batch_size=batch_size,
        total_n=total_n,
        partitioning=partitioning,
        seed=seed,
    )

    train_loss = train_fn_prox(
        model,
        trainloader,
        local_epochs,
        lr,
        device,
        gradient_clipping,
        dataset_name,
        weight_decay,
        mu,
        global_params,
        classification_type,
    )

    # Return FULL state dict for aggregation
    model_record = ArrayRecord(model.state_dict())

    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on the global test set."""

    dataset_name = context.run_config["dataset"]
    num_clients = context.run_config["num-clients"]
    non_iid_alpha = context.run_config.get("non-iid-alpha", 1.0)  # Default for extreme partitioning
    batch_size = context.run_config["batch-size"]
    total_n = context.run_config.get("total-n", 200)
    partition_id = context.node_config["partition-id"]
    classification_type = context.run_config.get("classification_type", "binary")
    partitioning = context.run_config.get("partitioning", "dirichlet")
    seed = context.run_config.get("seed", 42)

    model = get_model(dataset_name)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _, testloader = load_data(
        dataset_name=dataset_name,
        partition_id=partition_id,
        num_clients=num_clients,
        non_iid_alpha=non_iid_alpha,
        batch_size=batch_size,
        total_n=total_n,
        partitioning=partitioning,
        seed=seed,
    )

    eval_loss, eval_acc, eval_auc, eval_precision, eval_recall, eval_f1, eval_ap = test_fn(
        model,
        testloader,
        device,
        dataset_name,
        classification_type,
    )

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "eval_auc": eval_auc,
        "eval_precision": eval_precision,
        "eval_recall": eval_recall,
        "eval_f1": eval_f1,
        "eval_ap": eval_ap,
        "num-examples": len(testloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)




