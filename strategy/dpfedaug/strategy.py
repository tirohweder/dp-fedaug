import time
from logging import INFO
from pathlib import Path
from typing import Callable, Iterable, Optional
import os
import numpy as np
import torch
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.common import log, logger
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, Result

class DPFedAvg(FedAvg):
    """
    DP-FedAug Strategy:
    1. Handles DP-VAE generation and redistribution.
    2. Logs privacy budgets and generative quality to W&B.
    """
    def set_save_path_and_run_dir(self, path: Path, run_dir: str):
        self.save_path = path
        self.run_dir = run_dir

    def set_wandb_project(self, project_name: str):
        self.wandb_project_name = project_name

    def set_run_config(self, run_config: dict):
        self.wandb_run_config = run_config

    def _update_best_metric(self, current_round: int, metric: float, arrays: ArrayRecord) -> None:
        if metric > self.best_metric_so_far:
            self.best_metric_so_far = metric
            logger.log(INFO, f"💡 New best global model: AUC = {metric:.4f}")
            file_name = f"model_round_{current_round}_auc_{metric:.4f}.pth"
            torch.save(arrays.to_torch_state_dict(), self.save_path / file_name)

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        if not replies: return None, None
        # Filter out failed replies (those without content)
        valid_replies = [r for r in replies if r.has_content()]
        if not valid_replies: return None, None
        
        arrays_list = [reply.content["arrays"] for reply in valid_replies]
        metrics_list = [reply.content["metrics"] for reply in valid_replies if "metrics" in reply.content]
        if not arrays_list: return None, None
        
        state_dicts = [arr.to_torch_state_dict() for arr in arrays_list]
        weights = []
        for reply in valid_replies:
            metrics = reply.content.get("metrics")
            if metrics is None:
                weights.append(1.0)
            else:
                weights.append(float(metrics.get("num-examples", 1.0)))
        total_weight = sum(weights)
        if total_weight <= 0:
            return None, None

        avg_state_dict = {
            k: sum(sd[k] * w for sd, w in zip(state_dicts, weights)) / total_weight
            for k in state_dicts[0]
        }
        
        agg_metrics = {}
        if metrics_list:
            metric_weight_sum = sum(float(m.get("num-examples", 1.0)) for m in metrics_list)
            for k in metrics_list[0]:
                if k != "num-examples":
                    if metric_weight_sum > 0:
                        agg_metrics[k] = sum(float(m[k]) * float(m.get("num-examples", 1.0)) for m in metrics_list) / metric_weight_sum
                    else:
                        agg_metrics[k] = sum(float(m[k]) for m in metrics_list) / len(metrics_list)
        
        return ArrayRecord(avg_state_dict), MetricRecord(agg_metrics)

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        if not replies: return None
        valid_replies = [r for r in replies if r.has_content()]
        if not valid_replies: return None
        metrics_list = [reply.content["metrics"] for reply in valid_replies if "metrics" in reply.content]
        if not metrics_list: return None
        agg_metrics = {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in metrics_list[0] if k != "num-examples"}
        return MetricRecord(agg_metrics)

    def start(self, grid: Grid, initial_arrays: ArrayRecord, num_rounds: int, timeout: float = 3600, train_config: Optional[ConfigRecord] = None, evaluate_config: Optional[ConfigRecord] = None, evaluate_fn: Optional[Callable[[int, ArrayRecord], Optional[MetricRecord]]] = None) -> Result:
        cfg = self.wandb_run_config
        dataset_name = cfg["dataset"]
        total_n = cfg["total-n"]
        target_epsilon = cfg["target-epsilon"]  # "none" or numeric string like "1", "8"
        non_iid_alpha = cfg["non-iid-alpha"]
        seed = cfg["seed"]
        synthetic_count = cfg["synthetic-count"]
        
        # Format epsilon for display
        eps_display = "inf" if target_epsilon == "none" else target_epsilon
        
        # W&B Init
        wandb_enabled = False
        if os.getenv("WANDB_API_KEY") or os.getenv("WANDB_MODE") != "disabled":
            try:
                wandb.init(
                    project=getattr(self, 'wandb_project_name'),
                    name=f"DP_N{total_n}_eps{eps_display}_alpha{non_iid_alpha}_seed{seed}",
                    config=cfg,
                    tags=["DP-FedAug", dataset_name, f"eps_{eps_display}"]
                )
                wandb_enabled = True
            except: pass

        self.best_metric_so_far = 0.0
        epsilons_mean, epsilons_max = [], []

        # DP-FedAug Pre-processing
        if synthetic_count > 0:
            log(INFO, "🚀 [DP-FedAug] Starting DP generative phase...")
            node_ids = list(grid.get_node_ids())
            replies = grid.send_and_receive(
                messages=[Message(content=RecordDict({}), dst_node_id=n, message_type="query.fedaug_generate") for n in node_ids],
                timeout=3600
            )

            all_samples, all_labels = [], []
            empty_node_ids = set()
            for idx, r in enumerate(replies):
                node_id = getattr(r, "src_node_id", None)
                if node_id is None and idx < len(node_ids):
                    node_id = node_ids[idx]

                if not r.has_content():
                    raise RuntimeError(
                        "DP-FedAug Critical Failure: client returned no content "
                        f"(node_id={node_id})."
                    )
                if "status" in r.content:
                    status = r.content["status"]
                    try:
                        status_dict = status.to_dict()
                    except Exception:
                        status_dict = {}
                    if status_dict.get("success") is False:
                        reason = status_dict.get("reason", "unknown")
                        if reason == "empty_dataset":
                            empty_node_ids.add(node_id)
                            continue
                        raise RuntimeError(
                            "DP-FedAug Critical Failure: client failed in generative phase "
                            f"(node_id={node_id}, reason={reason})."
                        )
                if "samples" not in r.content or "labels" not in r.content:
                    raise RuntimeError(
                        "DP-FedAug Critical Failure: client missing samples/labels "
                        f"(node_id={node_id})."
                    )

                samples = r.content["samples"].to_numpy_ndarrays()[0]
                labels = r.content["labels"].to_numpy_ndarrays()[0]
                if samples.size == 0 or labels.size == 0:
                    if node_id is not None:
                        empty_node_ids.add(node_id)
                        continue
                    raise RuntimeError(
                        "DP-FedAug Critical Failure: client returned empty samples/labels "
                        f"(node_id={node_id})."
                    )

                all_samples.append(samples)
                all_labels.append(labels)
                if "epsilon_mean" in r.content:
                    epsilons_mean.append(r.content["epsilon_mean"].to_numpy_ndarrays()[0][0])
                if "epsilon_max" in r.content:
                    epsilons_max.append(r.content["epsilon_max"].to_numpy_ndarrays()[0][0])

            if wandb_enabled and epsilons_mean:
                log_data = {"Avg Client Epsilon (mean per-label)": np.mean(epsilons_mean)}
                if epsilons_max:
                    log_data["Avg Client Epsilon (max per-label, formal guarantee)"] = np.mean(epsilons_max)
                    log_data["Max Client Epsilon (formal guarantee)"] = np.max(epsilons_max)
                wandb.log(log_data)

            if all_samples:
                pool_x = np.concatenate(all_samples)
                pool_y = np.concatenate(all_labels)
                
                # Robustness: Check if we have enough data for all clients
                target_node_ids = [nid for nid in node_ids if nid not in empty_node_ids]
                total_needed = len(target_node_ids) * synthetic_count

                if not target_node_ids:
                    log(INFO, "DP-FedAug: All clients empty; skipping augmentation.")
                    target_node_ids = []
                
                if len(pool_x) < total_needed:
                    raise RuntimeError(
                        "DP-FedAug Critical Failure: insufficient synthetic samples generated. "
                        f"Expected {total_needed} (for {len(target_node_ids)} clients * {synthetic_count} samples), "
                        f"but only collected {len(pool_x)}. This usually means one or more clients failed "
                        "during the generative phase."
                    )

                # Shuffle available data
                idx = np.random.permutation(len(pool_x))
                pool_x, pool_y = pool_x[idx], pool_y[idx]

                per_client = synthetic_count
                store_msgs = []
                for i, node_id in enumerate(target_node_ids):
                    start, end = i * per_client, (i + 1) * per_client
                    content = RecordDict({"samples": ArrayRecord([pool_x[start:end]]), "labels": ArrayRecord([pool_y[start:end]])})
                    store_msgs.append(Message(content=content, dst_node_id=node_id, message_type="query.fedaug_store_data"))

                for node_id in empty_node_ids:
                    empty_samples = np.empty((0,) + pool_x.shape[1:], dtype=pool_x.dtype)
                    empty_labels = np.empty((0,), dtype=pool_y.dtype)
                    empty_content = RecordDict({
                        "samples": ArrayRecord([empty_samples]),
                        "labels": ArrayRecord([empty_labels])
                    })
                    store_msgs.append(Message(content=empty_content, dst_node_id=node_id, message_type="query.fedaug_store_data"))

                grid.send_and_receive(messages=store_msgs, timeout=300)
            else:
                if empty_node_ids and len(empty_node_ids) == len(node_ids):
                    log(INFO, "DP-FedAug: All clients empty; skipping augmentation.")
                else:
                    raise RuntimeError(
                        "DP-FedAug Critical Failure: all clients returned empty synthetic data. "
                        "This usually happens when one or more clients have empty partitions."
                    )

        # Main Loop
        arrays = initial_arrays
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            if res and wandb_enabled: wandb.log({"Global Test/AUC": res.get("auc"), "Communication Round": 0})

        for r in range(1, num_rounds + 1):
            train_replies = grid.send_and_receive(messages=self.configure_train(r, arrays, train_config or ConfigRecord(), grid), timeout=timeout)
            agg_arrays, agg_metrics = self.aggregate_train(r, train_replies)
            if agg_arrays: arrays = agg_arrays

            log_data = {"Communication Round": r}

            # Global evaluation
            if evaluate_fn:
                res = evaluate_fn(r, arrays)
                if res:
                    log_data.update({f"Global Test/{k}": v for k, v in dict(res).items()})

            # Per-client local evaluation for fairness metrics
            eval_config = ConfigRecord({"eval_local": True})
            eval_msgs = self.configure_evaluate(r, arrays, eval_config, grid)
            eval_replies = grid.send_and_receive(messages=eval_msgs, timeout=timeout)
            valid_eval = [rep for rep in eval_replies if rep.has_content() and "metrics" in rep.content]
            if valid_eval:
                client_accs = [float(rep.content["metrics"]["accuracy"]) for rep in valid_eval]
                log_data["Per-Client/Worst-Accuracy"] = min(client_accs)
                log_data["Per-Client/Best-Accuracy"] = max(client_accs)
                log_data["Per-Client/Mean-Accuracy"] = np.mean(client_accs)
                log_data["Per-Client/Std-Accuracy"] = np.std(client_accs)

            if wandb_enabled:
                wandb.log(log_data)

            # Always print round summary to stdout (visible even with W&B disabled)
            acc = log_data.get("Global Test/accuracy")
            loss = log_data.get("Global Test/loss")
            worst = log_data.get("Per-Client/Worst-Accuracy")
            best  = log_data.get("Per-Client/Best-Accuracy")
            acc_str  = f"acc={acc:.4f}" if acc  is not None else ""
            loss_str = f"loss={loss:.4f}" if loss is not None else ""
            fair_str = f"worst={worst:.4f} best={best:.4f}" if worst is not None else ""
            log(INFO, f"[Round {r:3d}] {acc_str}  {loss_str}  {fair_str}")

        if wandb_enabled:
            # Privacy accounting summary
            summary = {}
            if epsilons_mean:
                summary["Privacy/Synthetic-Epsilon-Mean"] = np.mean(epsilons_mean)
            if epsilons_max:
                summary["Privacy/Synthetic-Epsilon-Max"] = np.max(epsilons_max)

            updates_dp_enabled = cfg.get("updates-dp-enabled", False)
            if isinstance(updates_dp_enabled, str):
                updates_dp_enabled = updates_dp_enabled.lower() == "true"
            if updates_dp_enabled:
                summary["Privacy/Updates-DP-Epsilon"] = float(cfg.get("updates-dp-epsilon", 0))
                summary["Privacy/Updates-DP-Delta"] = float(cfg.get("updates-dp-delta", 0))

            if summary:
                wandb.log(summary)

            wandb.finish()
        return Result()
