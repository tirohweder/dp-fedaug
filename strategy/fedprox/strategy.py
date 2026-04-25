import os
from logging import INFO
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import wandb
from dotenv import load_dotenv
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log, logger
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info

load_dotenv()


class FedProxStrategy(FedAvg):
    """FedProx strategy with sample-weighted aggregation and W&B logging."""

    def set_save_path_and_run_dir(self, path: Path, run_dir: str):
        self.save_path = path
        self.run_dir = run_dir

    def set_wandb_project(self, project_name: str):
        self.wandb_project_name = project_name

    def set_run_config(self, run_config: dict):
        self.wandb_run_config = run_config

    def _update_best_metric(self, current_round: int, metric: float, arrays: ArrayRecord) -> None:
        """Update best AUC and save model checkpoint."""
        if metric > self.best_metric_so_far:
            self.best_metric_so_far = metric
            logger.log(INFO, f"💡 New best global model found: AUC = {metric:.4f}")
            file_name = f"model_round_{current_round}_auc_{metric:.4f}.pth"
            torch.save(arrays.to_torch_state_dict(), self.save_path / file_name)
            logger.log(INFO, f"💾 New best model saved to disk: {file_name}")

    # --- Sample-weighted aggregation ---
    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        if not replies:
            return None, None

        arrays_list = [reply.content["arrays"] for reply in replies]
        train_metrics_list = [reply.content["metrics"] for reply in replies if "metrics" in reply.content]

        if not arrays_list:
            return None, None

        state_dicts = [arr.to_torch_state_dict() for arr in arrays_list]
        avg_state_dict = {}
        weights = [float(reply.content["metrics"].get("num-examples", 1.0)) if "metrics" in reply.content else 1.0 for reply in replies]
        total_weight = sum(weights)
        if total_weight <= 0:
            return None, None

        for key in state_dicts[0].keys():
            avg_state_dict[key] = sum(sd[key] * w for sd, w in zip(state_dicts, weights)) / total_weight

        agg_arrays = ArrayRecord(avg_state_dict)

        agg_train_metrics = None
        if train_metrics_list:
            agg_train_metrics = {}
            metric_weight_sum = sum(float(m.get("num-examples", 1.0)) for m in train_metrics_list)
            for key in train_metrics_list[0].keys():
                if key != "num-examples":
                    if metric_weight_sum > 0:
                        agg_train_metrics[key] = sum(float(m[key]) * float(m.get("num-examples", 1.0)) for m in train_metrics_list) / metric_weight_sum
                    else:
                        values = [float(m[key]) for m in train_metrics_list]
                        agg_train_metrics[key] = sum(values) / len(values)
                else:
                    agg_train_metrics[key] = sum(m[key] for m in train_metrics_list)
            agg_train_metrics = MetricRecord(agg_train_metrics)

        return agg_arrays, agg_train_metrics

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        if not replies:
            return None

        eval_metrics_list = [reply.content["metrics"] for reply in replies if "metrics" in reply.content]
        if not eval_metrics_list:
            return None

        agg_eval_metrics = {}
        for key in eval_metrics_list[0].keys():
            if key != "num-examples":
                values = [m[key] for m in eval_metrics_list]
                agg_eval_metrics[key] = sum(values) / len(values)
            else:
                agg_eval_metrics[key] = sum(m[key] for m in eval_metrics_list)

        return MetricRecord(agg_eval_metrics)
    # --------------------------------------------------------

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[Callable[[int, ArrayRecord], Optional[MetricRecord]]] = None,
    ) -> Result:

        # --- W&B Initialization ---
        wandb_enabled = False
        try:
            # Try to login - works with WANDB_API_KEY env var OR ~/.netrc credentials
            wandb_api_key = os.getenv("WANDB_API_KEY")
            if wandb_api_key:
                wandb.login(key=wandb_api_key)
            else:
                # Try to use existing credentials from ~/.netrc (set by 'wandb login')
                wandb.login()

            cfg = self.wandb_run_config if hasattr(self, "wandb_run_config") else {}
            prox_mu = cfg.get("proximal_mu", 0.0)
            partitioning = cfg.get("partitioning", "dirichlet")
            total_n = cfg.get("total_n", 0)
            alpha = cfg.get("non_iid_alpha", "unk")
            seed = cfg.get("seed", "unk")

            # Build run name based on partitioning strategy
            if partitioning == "extreme":
                run_name = f"N{total_n}_extreme_mu{prox_mu}_seed{seed}"
            else:
                run_name = f"N{total_n}_alpha{alpha}_mu{prox_mu}_seed{seed}"

            project_name = getattr(self, "wandb_project_name", "FedProx-Baseline-Braintumor")

            # Build tags
            tags = ["FedProx", f"mu_{prox_mu}"]
            if partitioning == "extreme":
                tags.append("extreme")
            else:
                tags.append(f"alpha_{alpha}")

            wandb.init(
                project=project_name,
                name=run_name,
                config={"num_rounds": num_rounds, "strategy": "FedProx", **cfg},
                tags=tags,
            )
            wandb.define_metric("Communication Round")
            wandb.define_metric("Global Test/*", step_metric="Communication Round")
            wandb.define_metric("Training/*", step_metric="Communication Round")
            wandb_enabled = True
        except Exception as e:
            log(INFO, f"W&B initialization failed: {e}. Continuing without W&B logging.")

        self.best_metric_so_far = 0.0

        log_strategy_start_info(num_rounds, initial_arrays, train_config, evaluate_config)
        result = Result()
        arrays = initial_arrays

        # Initial evaluation
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            if res and wandb_enabled:
                initial_log = {"Communication Round": 0}
                for k, v in dict(res).items():
                    initial_log[f"Global Test/{k.replace('_', ' ').title()}"] = v
                wandb.log(initial_log)

        for current_round in range(1, num_rounds + 1):
            # Train
            train_replies = grid.send_and_receive(
                messages=self.configure_train(current_round, arrays, train_config or ConfigRecord(), grid),
                timeout=timeout,
            )
            agg_arrays, agg_train_metrics = self.aggregate_train(current_round, train_replies)
            if agg_arrays:
                arrays = agg_arrays

            if agg_train_metrics and wandb_enabled:
                train_log = {"Communication Round": current_round}
                for k, v in dict(agg_train_metrics).items():
                    if k != "num-examples":
                        train_log[f"Training/{k.replace('_', ' ').title()}"] = v
                wandb.log(train_log)

            # Global Eval
            if evaluate_fn:
                res = evaluate_fn(current_round, arrays)
                if res:
                    self._update_best_metric(current_round, res.get("auc", 0.0), arrays)
                    if wandb_enabled:
                        server_log = {"Communication Round": current_round}
                        for k, v in dict(res).items():
                            name = "AUC-ROC" if k == "auc" else "F1-Score" if k == "f1" else k.replace("_", " ").title()
                            server_log[f"Global Test/{name}"] = v
                        wandb.log(server_log)

        if wandb_enabled:
            wandb.finish()

        return result



