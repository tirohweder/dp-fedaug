from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy.stats as stats


def get_cfg_value(cfg: dict, *keys):
    for key in keys:
        if key in cfg:
            return cfg.get(key)
    return None


def to_float(value, *, zero_is_infinity: bool = True):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if zero_is_infinity and float(value) == 0.0:
            return math.inf
        return float(value)
    string_value = str(value).strip().lower()
    if string_value in {"none", "nan", ""}:
        return math.inf if zero_is_infinity else None
    if zero_is_infinity and string_value in {"0", "0.0"}:
        return math.inf
    if string_value in {"inf", "infinity"}:
        return math.inf
    try:
        return float(string_value)
    except ValueError:
        return None


def epsilon_label(value) -> str:
    if value is None or (isinstance(value, float) and (math.isinf(value) or np.isnan(value))):
        return "none"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def is_scalar_wandb_value(value) -> bool:
    return isinstance(value, (int, float, str, bool, np.number)) or value is None


def flatten_wandb_value(value):
    if is_scalar_wandb_value(value):
        return value
    if isinstance(value, (list, tuple)) and len(value) <= 10:
        return json.dumps(value)
    if isinstance(value, dict) and len(value) <= 10:
        return json.dumps(value, sort_keys=True)
    return None


def load_or_fetch_dataframe(
    cache_path: Path,
    *,
    refresh: bool,
    fetch_fn: Callable[[], pd.DataFrame],
    normalize_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path)
    else:
        df = fetch_fn()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    if normalize_fn is not None:
        df = normalize_fn(df)
    return df


def fetch_seeded_runs(project: str, entity: str | None, timeout: int) -> pd.DataFrame:
    import wandb

    api = wandb.Api(timeout=timeout)
    path = f"{entity}/{project}" if entity else project
    rows = []

    for run in api.runs(path):
        cfg = run.config
        summary = run.summary
        accuracy_key = (
            "Global Test/accuracy (Max)"
            if "Global Test/accuracy (Max)" in summary
            else "Global Test/accuracy"
        )
        rows.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "project": run.project,
                "entity": getattr(run, "entity", None),
                "total_n": get_cfg_value(cfg, "total-n", "total_n"),
                "synthetic_count": get_cfg_value(cfg, "synthetic-count", "synthetic_count"),
                "alpha": get_cfg_value(cfg, "non-iid-alpha", "non_iid_alpha", "alpha"),
                "partitioning": get_cfg_value(cfg, "partitioning"),
                "balancing": get_cfg_value(cfg, "balancing"),
                "seed": get_cfg_value(cfg, "seed"),
                "target_epsilon": get_cfg_value(cfg, "target-epsilon", "target_epsilon", "epsilon"),
                "updates_dp_enabled": get_cfg_value(cfg, "updates-dp-enabled", "updates_dp_enabled"),
                "updates_dp_epsilon": get_cfg_value(cfg, "updates-dp-epsilon", "updates_dp_epsilon"),
                "accuracy": summary.get(accuracy_key),
                "auc": summary.get("Global Test/auc") or summary.get("Global Test/AUC"),
                "epsilon_achieved_val": summary.get("Avg Client Epsilon"),
            }
        )

    return pd.DataFrame(rows)


def normalize_seeded_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_n"] = pd.to_numeric(df["total_n"], errors="coerce")
    df["synthetic_count"] = pd.to_numeric(df["synthetic_count"], errors="coerce").fillna(0)
    df["updates_dp_enabled"] = df["updates_dp_enabled"].fillna(False).astype(bool)
    df["updates_dp_epsilon"] = df["updates_dp_epsilon"].apply(to_float)
    df["alpha"] = df["alpha"].astype(str)
    partitioning = df["partitioning"].astype(str).str.lower()
    df.loc[partitioning.isin({"extreme", "1"}), "alpha"] = "extreme"
    df["alpha"] = df["alpha"].replace({"inf": "Infinity"})
    df["target_epsilon_num"] = df["target_epsilon"].apply(to_float)
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df["target_epsilon_label"] = df["target_epsilon_num"].apply(epsilon_label)
    df["updates_dp_epsilon_label"] = df["updates_dp_epsilon"].apply(epsilon_label)
    return df


def summarize_seeded_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "total_n",
        "synthetic_count",
        "alpha",
        "target_epsilon_label",
        "updates_dp_enabled",
        "updates_dp_epsilon_label",
    ]
    return (
        df.groupby(group_cols, dropna=False)
        .agg(
            runs=("accuracy", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
        )
        .reset_index()
    )


def fetch_fedprox_runs(project: str, entity: str | None, timeout: int) -> pd.DataFrame:
    import wandb

    api = wandb.Api(timeout=timeout)
    path = f"{entity}/{project}" if entity else project
    rows = []

    for run in api.runs(path):
        cfg = run.config
        summary = run.summary
        row = {
            "run_id": run.id,
            "run_name": run.name,
            "project": run.project,
            "entity": getattr(run, "entity", None),
            "state": run.state,
            "created_at": run.created_at,
            "dataset": get_cfg_value(cfg, "dataset"),
            "total_n": get_cfg_value(cfg, "total-n", "total_n"),
            "partitioning": get_cfg_value(cfg, "partitioning"),
            "alpha": get_cfg_value(cfg, "non-iid-alpha", "non_iid_alpha", "alpha"),
            "proximal_mu": get_cfg_value(cfg, "proximal-mu", "proximal_mu", "mu"),
            "seed": get_cfg_value(cfg, "seed"),
        }
        for key in [
            "Global Test/accuracy",
            "Global Test/Accuracy",
            "Global Test/accuracy (Max)",
            "Global Test/Accuracy (Max)",
            "Global Test/auc",
            "Global Test/Auc",
            "Global Test/AUC",
            "Global Test/AUC-ROC",
            "Global Test/loss",
            "Global Test/Loss",
            "Global Test/f1",
            "Global Test/F1",
            "Global Test/F1-Score",
            "Global Test/precision",
            "Global Test/Precision",
            "Global Test/recall",
            "Global Test/Recall",
            "Global Test/ap",
            "Global Test/Ap",
            "Communication Round",
        ]:
            if key in summary:
                row[key] = summary.get(key)
        rows.append(row)

    return pd.DataFrame(rows)


def normalize_fedprox_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_n"] = pd.to_numeric(df["total_n"], errors="coerce")
    df["alpha"] = df["alpha"].apply(lambda value: to_float(value, zero_is_infinity=False))
    df["proximal_mu"] = pd.to_numeric(df["proximal_mu"], errors="coerce")

    accuracy_columns = [
        "Global Test/accuracy (Max)",
        "Global Test/Accuracy (Max)",
        "Global Test/accuracy",
        "Global Test/Accuracy",
    ]
    df["accuracy"] = np.nan
    for column in accuracy_columns:
        if column in df.columns:
            df["accuracy"] = pd.to_numeric(df[column], errors="coerce").combine_first(df["accuracy"])

    auc = df.get("Global Test/auc")
    if auc is None:
        auc = df.get("Global Test/Auc")
    if auc is None:
        auc = df.get("Global Test/AUC")
    if auc is None:
        auc = df.get("Global Test/AUC-ROC")
    df["auc"] = pd.to_numeric(auc, errors="coerce")
    return df


def summarize_fedprox_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["total_n", "partitioning", "alpha", "proximal_mu"]
    return (
        df.groupby(group_cols, dropna=False)
        .agg(
            runs=("accuracy", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            auc_mean=("auc", "mean"),
        )
        .reset_index()
    )


def fetch_exploration_runs(project: str, entity: str | None, timeout: int) -> pd.DataFrame:
    import wandb

    api = wandb.Api(timeout=timeout)
    path = f"{entity}/{project}" if entity else project
    rows = []

    for run in api.runs(path):
        row = {"run_id": run.id, "run_name": run.name, "state": run.state}
        for key, value in run.config.items():
            if key.startswith("_"):
                continue
            flattened = flatten_wandb_value(value)
            if flattened is not None:
                row[key] = flattened
        for key, value in run.summary.items():
            flattened = flatten_wandb_value(value)
            if flattened is not None:
                row[key] = flattened
        rows.append(row)

    return pd.DataFrame(rows)


def parse_exploration_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def parse_name(name: str) -> dict:
        result = {
            "experiment_type": None,
            "has_dropout": None,
            "target_epsilon": None,
            "num_epochs": None,
            "run_number": None,
        }

        if name.startswith("epsilon_"):
            result["experiment_type"] = "epsilon"
            if "_no_dropout_" in name:
                result["has_dropout"] = False
            elif "_with_dropout_" in name:
                result["has_dropout"] = True

            eps_match = re.search(r"_eps_(\d+|None)_", name)
            if eps_match:
                eps_value = eps_match.group(1)
                result["target_epsilon"] = None if eps_value == "None" else int(eps_value)

        elif name.startswith("epochs_"):
            result["experiment_type"] = "epochs"
            if "_no_dropout_" in name:
                result["has_dropout"] = False
            elif "_with_dropout_" in name:
                result["has_dropout"] = True

            epoch_match = re.search(r"_epochs_(\d+)_", name)
            if epoch_match:
                result["num_epochs"] = int(epoch_match.group(1))

        run_match = re.search(r"_run_(\d+)$", name)
        if run_match:
            result["run_number"] = int(run_match.group(1))

        return result

    parsed = df["run_name"].apply(parse_name).apply(pd.Series)
    for column in parsed.columns:
        df[column] = parsed[column]

    if "val_acc" in df.columns:
        df["accuracy"] = pd.to_numeric(df["val_acc"], errors="coerce")
    elif "accuracy" in df.columns:
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")

    return df


def compute_group_summary(df: pd.DataFrame, group_cols: list[str], value_col: str = "accuracy") -> pd.DataFrame:
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            accuracy_mean=(value_col, "mean"),
            accuracy_std=(value_col, "std"),
            accuracy_count=(value_col, "count"),
        )
        .reset_index()
    )
    summary["accuracy_sem"] = summary["accuracy_std"] / np.sqrt(summary["accuracy_count"])
    summary["accuracy_ci95"] = summary.apply(
        lambda row: stats.t.ppf(0.975, row["accuracy_count"] - 1) * row["accuracy_sem"]
        if row["accuracy_count"] > 1
        else 0,
        axis=1,
    )
    return summary
