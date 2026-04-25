"""
DP-FedAug experiment runner for CIFAR-10.

Mirrors the MNIST DP-FedAug runner:
- supports local execution and HPC script generation
- avoids illogical config combinations
- can optionally query W&B and keep only missing runs
"""

import argparse
import os
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

# Suppress warnings at import time
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# CONFIGURATION - Edit these values directly (no command line args required)
# =============================================================================

# Execution mode: "local" or "hpc"
EXECUTION_MODE = "hpc"

# HPC Configuration (only used when execution mode = "hpc").
# Override via env vars in your .env file (see .env.example).
HPC_HOST = os.environ.get("HPC_HOST", "user@hpc.example.com")
HPC_PROJECT_DIR = os.environ.get("HPC_PROJECT_DIR", "~/dp-fedaug")
HPC_QUEUE = "gpuv100"
HPC_WALLTIME = "24:00"
HPC_GPUS = 1
HPC_CORES = 4
HPC_MEM = "64GB"

# Experiment parameters
STARTING_SEED = 42
NUM_SEEDS = 3
DRY_RUN = False

# DP-FedAug study grid
SYNTHETIC_COUNTS = [0, 100, 500]
TARGET_EPSILON_VALUES = [None, 8.0, 4.0, 1.0]
ALPHA_VALUES = [0.1, float("inf")]
TRAIN_SIZES = [5000, 10000]

# DP on model updates (client-side LocalDpMod)
UPDATES_DP_ENABLED_VALUES = [False]
UPDATES_DP_EPSILON_VALUES = [1.0]
UPDATES_DP_DELTA_VALUES = [1e-5]
UPDATES_DP_CLIPPING_NORMS = [1.0]
UPDATES_DP_SENSITIVITIES = [1.0]

# Training configuration
NUM_CLIENTS = 10
NUM_ROUNDS = 50
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
IMG_SIZE = 32

# Synthetic training configuration
SYNTHETIC_EPOCHS = 40
SYNTHETIC_BATCH_SIZE = 512
SYNTHETIC_LATENT_DIM = 96
SYNTHETIC_KL_WARMUP = 10
SYNTHETIC_LR = 0.001
SYNTHETIC_DELTA = 1e-5
MAX_GRAD_NORM = 1.0
SYNTHETIC_EVAL_METRICS = False

# Partitioning
PARTITIONING_STRATEGIES = ["extreme", "dirichlet"]

# Logging / metadata
WANDB_PROJECT = "DP-FedAug-CIFAR10-D2P-Study-seeded"
WANDB_ENTITY = "rohwedertimm"
CLASSIFICATION_TYPE = "multiclass"
BALANCING = "scaled"


# =============================================================================
# HELPERS
# =============================================================================

def ensure_dataset_cached() -> None:
    """Pre-download CIFAR-10 dataset if not already cached."""
    cache_dir = "data/cifar10"
    if not os.path.exists(cache_dir) or not os.listdir(cache_dir):
        print("Downloading CIFAR-10 dataset (one-time only)...")
        try:
            from datasets import load_dataset

            load_dataset("cifar10", cache_dir=cache_dir)
            print("CIFAR-10 dataset cached successfully.\n")
        except Exception as exc:
            print(f"Could not pre-cache dataset: {exc}")
            print("Will download during first run.\n")
    else:
        print("CIFAR-10 dataset already cached.\n")


def modify_pyproject_for_dpfedaug() -> str:
    """Modify pyproject.toml to use DP-FedAug strategy."""
    pyproject_path = Path("pyproject.toml")
    original_content = pyproject_path.read_text()

    modified_content = original_content
    for strategy in ["fedaug", "fedavg", "fedprox"]:
        modified_content = modified_content.replace(
            f'serverapp = "strategy.{strategy}.server_app:app"',
            'serverapp = "strategy.dpfedaug.server_app:app"',
        ).replace(
            f'clientapp = "strategy.{strategy}.client_app:app"',
            'clientapp = "strategy.dpfedaug.client_app:app"',
        )

    pyproject_path.write_text(modified_content)
    return original_content


def restore_pyproject(content: str) -> None:
    """Restore original pyproject.toml content."""
    Path("pyproject.toml").write_text(content)


def get_env_vars() -> dict[str, str]:
    """Get environment variables for subprocess."""
    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env["GRPC_VERBOSITY"] = "ERROR"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["HF_HUB_VERBOSITY"] = "error"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["DATASETS_VERBOSITY"] = "error"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_SILENT"] = "true"
    env["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::FutureWarning,ignore::UserWarning"
    env["RAY_DEDUP_LOGS"] = "1"
    env["TF_ENABLE_ONEDNN_OPTS"] = "0"
    env["KERAS_BACKEND"] = "torch"
    return env


def _format_epsilon_label(value: float | None) -> str:
    if value is None:
        return "none"
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def _format_alpha_key(value: float | None) -> str | None:
    if value is None:
        return None
    if value == float("inf"):
        return "inf"
    if float(value).is_integer():
        return f"{value:.1f}"
    return str(value)


def _format_partition_info(exp: dict[str, Any]) -> str:
    if exp["partitioning"] == "extreme":
        return "Part=extreme"
    alpha = exp.get("alpha")
    alpha_str = "inf" if alpha == float("inf") else str(alpha)
    return f"Part=dirichlet(a={alpha_str})"


def _format_updates_dp_info(exp: dict[str, Any]) -> str:
    if not exp["updates_dp_enabled"]:
        return "off"
    return (
        f"on (e={exp['updates_dp_epsilon']}, d={exp['updates_dp_delta']}, "
        f"clip={exp['updates_dp_clipping_norm']}, sens={exp['updates_dp_sensitivity']})"
    )


def _get_cfg_value(cfg: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in cfg:
            return cfg.get(key)
    return None


def _normalize_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "none", "nan"}:
            return None
        if lowered in {"inf", "infinity"}:
            return float("inf")
    return float(value)


def _normalize_target_epsilon(value: Any) -> float | None:
    normalized = _normalize_optional_float(value)
    if normalized is None:
        return None
    if normalized in {0.0, float("inf")}:
        return None
    return normalized


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _normalize_alpha(value: Any, partitioning: str) -> float | None:
    if partitioning == "extreme":
        return None
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "extreme":
            return None
        if lowered in {"inf", "infinity"}:
            return float("inf")
    return float(value)


def _is_logical_experiment(exp: dict[str, Any]) -> bool:
    if exp["synthetic_count"] == 0 and exp["target_epsilon"] is not None:
        return False
    if not exp["updates_dp_enabled"]:
        invalid_updates_fields = (
            exp.get("updates_dp_epsilon"),
            exp.get("updates_dp_delta"),
            exp.get("updates_dp_clipping_norm"),
            exp.get("updates_dp_sensitivity"),
        )
        if any(value is not None for value in invalid_updates_fields):
            return False
    if exp["partitioning"] == "extreme" and exp.get("alpha") is not None:
        return False
    if exp["partitioning"] == "dirichlet" and exp.get("alpha") is None:
        return False
    return True


def _make_run_key(exp: dict[str, Any]) -> tuple[Any, ...]:
    alpha_key = "extreme"
    if exp["partitioning"] == "dirichlet":
        alpha_key = _format_alpha_key(exp.get("alpha"))

    return (
        int(exp["total_n"]),
        exp["partitioning"],
        alpha_key,
        int(exp["synthetic_count"]),
        _format_epsilon_label(exp["target_epsilon"]),
        exp["balancing"],
        bool(exp["updates_dp_enabled"]),
        _format_epsilon_label(exp["updates_dp_epsilon"]) if exp["updates_dp_enabled"] else "none",
        exp["updates_dp_delta"] if exp["updates_dp_enabled"] else None,
        exp["updates_dp_clipping_norm"] if exp["updates_dp_enabled"] else None,
        exp["updates_dp_sensitivity"] if exp["updates_dp_enabled"] else None,
        int(exp["seed"]),
    )


def build_config_parts(
    target_epsilon: float | None,
    total_n: int,
    synthetic_count: int,
    partitioning: str,
    seed: int,
    balancing: str,
    wandb_project: str,
    updates_dp_enabled: bool,
    updates_dp_epsilon: float | None = None,
    updates_dp_delta: float | None = None,
    updates_dp_clipping_norm: float | None = None,
    updates_dp_sensitivity: float | None = None,
    alpha: float | None = None,
    for_bash: bool = False,
) -> list[str]:
    """Build the Flower run config parts."""
    q = "'" if for_bash else '"'
    target_eps_str = "none" if target_epsilon is None else str(target_epsilon)

    def format_value(value: Any) -> str:
        if isinstance(value, str):
            return f"{q}{value}{q}"
        return str(value)

    config_parts = [
        f"dataset={q}cifar10{q}",
        f"num-clients={NUM_CLIENTS}",
        f"target-epsilon={q}{target_eps_str}{q}",
        f"synthetic-count={synthetic_count}",
        f"synthetic-epochs={SYNTHETIC_EPOCHS}",
        f"synthetic-batch-size={SYNTHETIC_BATCH_SIZE}",
        f"synthetic-latent-dim={SYNTHETIC_LATENT_DIM}",
        f"synthetic-kl-warmup={SYNTHETIC_KL_WARMUP}",
        f"synthetic-lr={SYNTHETIC_LR}",
        f"synthetic-delta={SYNTHETIC_DELTA}",
        f"max-grad-norm={MAX_GRAD_NORM}",
        f"synthetic-eval-metrics={str(SYNTHETIC_EVAL_METRICS).lower()}",
        f"img-size={IMG_SIZE}",
        f"num-server-rounds={NUM_ROUNDS}",
        f"num-local-epochs={LOCAL_EPOCHS}",
        f"lr={LEARNING_RATE}",
        f"batch-size={BATCH_SIZE}",
        f"total-n={total_n}",
        f"partitioning={q}{partitioning}{q}",
        f"balancing={q}{balancing}{q}",
        f"wandb-project={q}{wandb_project}{q}",
        f"seed={seed}",
        f"gradient_clipping=true",
        f"classification_type={q}{CLASSIFICATION_TYPE}{q}",
        f"weight-decay={WEIGHT_DECAY}",
        f"updates-dp-enabled={str(updates_dp_enabled).lower()}",
        f"updates-dp-epsilon={format_value(updates_dp_epsilon if updates_dp_enabled else 'none')}",
        f"updates-dp-delta={format_value(updates_dp_delta if updates_dp_enabled else 'none')}",
        f"updates-dp-clipping-norm={format_value(updates_dp_clipping_norm if updates_dp_enabled else 'none')}",
        f"updates-dp-sensitivity={format_value(updates_dp_sensitivity if updates_dp_enabled else 'none')}",
    ]

    if partitioning == "dirichlet" and alpha is not None:
        alpha_str = "inf" if alpha == float("inf") else str(alpha)
        if alpha == float("inf"):
            config_parts.insert(2, f"non-iid-alpha={q}{alpha_str}{q}")
        else:
            config_parts.insert(2, f"non-iid-alpha={alpha_str}")
    elif partitioning == "extreme":
        config_parts.insert(2, f"non-iid-alpha={q}extreme{q}")

    return config_parts


def run_experiment_local(
    target_epsilon: float | None,
    total_n: int,
    synthetic_count: int,
    partitioning: str,
    seed: int,
    balancing: str,
    wandb_project: str,
    updates_dp_enabled: bool,
    updates_dp_epsilon: float | None = None,
    updates_dp_delta: float | None = None,
    updates_dp_clipping_norm: float | None = None,
    updates_dp_sensitivity: float | None = None,
    alpha: float | None = None,
) -> None:
    """Run a single experiment locally."""
    original_pyproject = modify_pyproject_for_dpfedaug()
    env = get_env_vars()

    try:
        config_parts = build_config_parts(
            target_epsilon=target_epsilon,
            total_n=total_n,
            synthetic_count=synthetic_count,
            partitioning=partitioning,
            seed=seed,
            balancing=balancing,
            wandb_project=wandb_project,
            updates_dp_enabled=updates_dp_enabled,
            updates_dp_epsilon=updates_dp_epsilon,
            updates_dp_delta=updates_dp_delta,
            updates_dp_clipping_norm=updates_dp_clipping_norm,
            updates_dp_sensitivity=updates_dp_sensitivity,
            alpha=alpha,
        )
        cmd = ["flwr", "run", ".", "--run-config", " ".join(config_parts)]

        exp = {
            "partitioning": partitioning,
            "alpha": alpha,
            "target_epsilon": target_epsilon,
            "synthetic_count": synthetic_count,
            "updates_dp_enabled": updates_dp_enabled,
            "updates_dp_epsilon": updates_dp_epsilon,
            "updates_dp_delta": updates_dp_delta,
            "updates_dp_clipping_norm": updates_dp_clipping_norm,
            "updates_dp_sensitivity": updates_dp_sensitivity,
        }

        eps_display = "inf (No DP)" if target_epsilon is None else f"{target_epsilon}"
        print(f"\n{'=' * 70}")
        print(
            f"DP-FedAug-CIFAR10: N={total_n} | {_format_partition_info(exp)} | "
            f"e={eps_display} | Synth={synthetic_count} | "
            f"Updates-DP={_format_updates_dp_info(exp)} | Seed={seed}"
        )
        print(f"{'=' * 70}")

        start_time = time.time()
        subprocess.run(cmd, check=False, env=env)
        duration = time.time() - start_time
        print(f"Finished in {duration:.1f}s")
    finally:
        restore_pyproject(original_pyproject)


def generate_lsf_script(experiments: list[dict[str, Any]]) -> str:
    """Generate an LSF batch script for DTU HPC."""
    experiment_cmds: list[str] = []
    for exp in experiments:
        config_parts = build_config_parts(
            target_epsilon=exp["target_epsilon"],
            total_n=exp["total_n"],
            synthetic_count=exp["synthetic_count"],
            partitioning=exp["partitioning"],
            seed=exp["seed"],
            balancing=exp["balancing"],
            wandb_project=exp["wandb_project"],
            updates_dp_enabled=exp["updates_dp_enabled"],
            updates_dp_epsilon=exp.get("updates_dp_epsilon"),
            updates_dp_delta=exp.get("updates_dp_delta"),
            updates_dp_clipping_norm=exp.get("updates_dp_clipping_norm"),
            updates_dp_sensitivity=exp.get("updates_dp_sensitivity"),
            alpha=exp.get("alpha"),
            for_bash=True,
        )
        experiment_cmds.append(f'flwr run . --run-config "{" ".join(config_parts)}"')

    script = f"""#!/bin/bash
#BSUB -J dpfedaug-cifar10
#BSUB -q {HPC_QUEUE}
#BSUB -W {HPC_WALLTIME}
#BSUB -n {HPC_CORES}
#BSUB -R "rusage[mem={HPC_MEM}]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num={HPC_GPUS}:mode=exclusive_process"
#BSUB -o logs/dpfedaug_cifar10_%J.out
#BSUB -e logs/dpfedaug_cifar10_%J.err

module load python3/3.12.4
module load cuda/12.1

cd {HPC_PROJECT_DIR}
mkdir -p logs

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Activated venv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Activated .venv/bin/activate"
else
    echo "ERROR: No virtual environment found!"
    exit 1
fi

if ! command -v flwr &> /dev/null; then
    echo "ERROR: flwr command not found!"
    exit 1
fi
echo "flwr version: $(flwr --version)"

export TF_CPP_MIN_LOG_LEVEL=3
export GRPC_VERBOSITY=ERROR
export WANDB_SILENT=true
export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::FutureWarning,ignore::UserWarning"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_VERBOSITY=error
export DATASETS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=1
export TF_ENABLE_ONEDNN_OPTS=0
export KERAS_BACKEND=torch

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
    echo "Loaded .env"
else
    echo "WARN: .env not found"
fi

if [ -n "$WANDB_API_KEY" ]; then
    export WANDB_MODE=online
else
    export WANDB_MODE=offline
fi

python -c "from datasets import load_dataset; load_dataset('cifar10', cache_dir='data/cifar10')" 2>/dev/null || true

python -c "
from pathlib import Path
p = Path('pyproject.toml')
c = p.read_text()
for s in ['fedaug', 'fedavg', 'fedprox']:
    c = c.replace(f'serverapp = \\"strategy.{{s}}.server_app:app\\"', 'serverapp = \\"strategy.dpfedaug.server_app:app\\"')
    c = c.replace(f'clientapp = \\"strategy.{{s}}.client_app:app\\"', 'clientapp = \\"strategy.dpfedaug.client_app:app\\"')
p.write_text(c)
"

echo "Starting DP-FedAug CIFAR-10 experiments..."
echo "Total experiments: {len(experiment_cmds)}"
echo ""

EXPERIMENT_NUM=0
"""

    for idx, cmd in enumerate(experiment_cmds):
        exp = experiments[idx]
        eps_display = "inf (No DP)" if exp["target_epsilon"] is None else f"{exp['target_epsilon']}"
        script += f"""
EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))
echo "========================================"
echo "[$EXPERIMENT_NUM/{len(experiment_cmds)}] N={exp['total_n']} | {_format_partition_info(exp)} | e={eps_display} | Synth={exp['synthetic_count']} | Updates-DP={_format_updates_dp_info(exp)} | Seed={exp['seed']}"
echo "========================================"
{cmd}
"""

    script += """
echo ""
echo "All experiments completed!"
"""
    return script


def submit_to_hpc(experiments: list[dict[str, Any]]) -> None:
    """Generate LSF script and copy it to DTU HPC."""
    lsf_script = generate_lsf_script(experiments)

    local_script_path = Path("hpc/cifar_dpfedaug_hpc.sh")
    local_script_path.parent.mkdir(parents=True, exist_ok=True)
    local_script_path.write_text(lsf_script, newline="\n")
    print(f"Generated LSF script: {local_script_path}")

    print(f"\nCopying to HPC ({HPC_HOST})...")
    try:
        scp_cmd = ["scp", str(local_script_path), f"{HPC_HOST}:{HPC_PROJECT_DIR}/hpc/"]
        subprocess.run(scp_cmd, check=True)
        print("Script copied to HPC successfully!")
        print("")
        print("=" * 50)
        print("TO SUBMIT THE JOB, run these commands:")
        print("=" * 50)
        print(f"  ssh {HPC_HOST}")
        print(f"  cd {HPC_PROJECT_DIR}")
        print("  bsub < hpc/cifar_dpfedaug_hpc.sh")
        print("=" * 50)
    except subprocess.CalledProcessError as exc:
        print(f"Error copying to HPC: {exc}")
        print(f"LSF script saved locally at: {local_script_path}")


def iter_updates_dp_configs() -> list[tuple[bool, float | None, float | None, float | None, float | None]]:
    configs: list[tuple[bool, float | None, float | None, float | None, float | None]] = []
    for enabled in UPDATES_DP_ENABLED_VALUES:
        if not enabled:
            configs.append((False, None, None, None, None))
            continue
        for epsilon in UPDATES_DP_EPSILON_VALUES:
            for delta in UPDATES_DP_DELTA_VALUES:
                for clipping_norm in UPDATES_DP_CLIPPING_NORMS:
                    for sensitivity in UPDATES_DP_SENSITIVITIES:
                        configs.append((True, epsilon, delta, clipping_norm, sensitivity))
    return configs


def collect_experiments(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Collect all logical experiment configurations."""
    experiments: list[dict[str, Any]] = []

    for total_n in TRAIN_SIZES:
        for partitioning in args.partitioning:
            alpha_values = [None] if partitioning == "extreme" else ALPHA_VALUES
            for alpha in alpha_values:
                for synthetic_count in SYNTHETIC_COUNTS:
                    eps_values = [None] if synthetic_count == 0 else TARGET_EPSILON_VALUES
                    for target_epsilon in eps_values:
                        for (
                            updates_dp_enabled,
                            updates_dp_epsilon,
                            updates_dp_delta,
                            updates_dp_clipping_norm,
                            updates_dp_sensitivity,
                        ) in iter_updates_dp_configs():
                            for seed_idx in range(args.num_seeds):
                                seed = args.starting_seed + seed_idx
                                exp = {
                                    "target_epsilon": target_epsilon,
                                    "total_n": total_n,
                                    "synthetic_count": synthetic_count,
                                    "partitioning": partitioning,
                                    "seed": seed,
                                    "balancing": args.balancing,
                                    "wandb_project": args.wandb_project,
                                    "updates_dp_enabled": updates_dp_enabled,
                                    "updates_dp_epsilon": updates_dp_epsilon,
                                    "updates_dp_delta": updates_dp_delta,
                                    "updates_dp_clipping_norm": updates_dp_clipping_norm,
                                    "updates_dp_sensitivity": updates_dp_sensitivity,
                                }
                                if alpha is not None:
                                    exp["alpha"] = alpha
                                if _is_logical_experiment(exp):
                                    experiments.append(exp)

    return experiments


def _run_key_from_wandb_config(cfg: dict[str, Any]) -> tuple[Any, ...] | None:
    partitioning_raw = _get_cfg_value(cfg, "partitioning")
    partitioning = str(partitioning_raw).strip().lower() if partitioning_raw is not None else "dirichlet"

    total_n_raw = _get_cfg_value(cfg, "total-n", "total_n")
    synthetic_count_raw = _get_cfg_value(cfg, "synthetic-count", "synthetic_count")
    seed_raw = _get_cfg_value(cfg, "seed")
    if total_n_raw is None or synthetic_count_raw is None or seed_raw is None:
        return None

    updates_dp_enabled = _normalize_bool(_get_cfg_value(cfg, "updates-dp-enabled", "updates_dp_enabled"))
    exp = {
        "total_n": int(float(total_n_raw)),
        "partitioning": partitioning,
        "alpha": _normalize_alpha(
            _get_cfg_value(cfg, "non-iid-alpha", "non_iid_alpha", "alpha"),
            partitioning,
        ),
        "synthetic_count": int(float(synthetic_count_raw)),
        "seed": int(float(seed_raw)),
        "balancing": str(_get_cfg_value(cfg, "balancing") or BALANCING),
        "target_epsilon": _normalize_target_epsilon(
            _get_cfg_value(cfg, "target-epsilon", "target_epsilon", "epsilon")
        ),
        "updates_dp_enabled": updates_dp_enabled,
        "updates_dp_epsilon": _normalize_optional_float(
            _get_cfg_value(cfg, "updates-dp-epsilon", "updates_dp_epsilon")
        ) if updates_dp_enabled else None,
        "updates_dp_delta": _normalize_optional_float(
            _get_cfg_value(cfg, "updates-dp-delta", "updates_dp_delta")
        ) if updates_dp_enabled else None,
        "updates_dp_clipping_norm": _normalize_optional_float(
            _get_cfg_value(cfg, "updates-dp-clipping-norm", "updates_dp_clipping_norm")
        ) if updates_dp_enabled else None,
        "updates_dp_sensitivity": _normalize_optional_float(
            _get_cfg_value(cfg, "updates-dp-sensitivity", "updates_dp_sensitivity")
        ) if updates_dp_enabled else None,
    }

    if not _is_logical_experiment(exp):
        return None
    return _make_run_key(exp)


def fetch_completed_run_keys_from_wandb(
    entity: str | None,
    project: str,
    timeout: int,
) -> set[tuple[Any, ...]]:
    """Fetch completed W&B runs and return normalized run keys."""
    import wandb

    api = wandb.Api(timeout=timeout)
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path)

    completed_keys: set[tuple[Any, ...]] = set()
    for run in runs:
        if getattr(run, "state", "").lower() != "finished":
            continue
        key = _run_key_from_wandb_config(run.config or {})
        if key is not None:
            completed_keys.add(key)
    return completed_keys


def filter_missing_experiments_from_wandb(
    experiments: list[dict[str, Any]],
    entity: str | None,
    project: str,
    timeout: int,
) -> tuple[list[dict[str, Any]], int]:
    """Keep only experiment configs that are not yet present in W&B."""
    completed_keys = fetch_completed_run_keys_from_wandb(entity=entity, project=project, timeout=timeout)
    filtered = [exp for exp in experiments if _make_run_key(exp) not in completed_keys]
    return filtered, len(experiments) - len(filtered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIFAR DP-FedAug Experiments")
    parser.add_argument("--num-seeds", type=int, default=NUM_SEEDS, help="Number of seeds per configuration.")
    parser.add_argument("--starting-seed", type=int, default=STARTING_SEED, help="Starting seed value.")
    parser.add_argument("--dry-run", action="store_true", help="Print configurations without running.")
    parser.add_argument(
        "--partitioning",
        type=str,
        nargs="+",
        default=PARTITIONING_STRATEGIES,
        choices=["dirichlet", "extreme"],
        help="Partitioning strategies to test.",
    )
    parser.add_argument(
        "--balancing",
        type=str,
        default=BALANCING,
        choices=["none", "scaled"],
        help="Synthetic balancing strategy.",
    )
    parser.add_argument(
        "--execution-mode",
        type=str,
        default=EXECUTION_MODE,
        choices=["local", "hpc"],
        help="Execution mode override.",
    )
    parser.add_argument(
        "--resume-missing-wandb",
        action="store_true",
        help="Query W&B and keep only configurations that do not already have a finished run.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=WANDB_ENTITY,
        help="W&B entity/user for --resume-missing-wandb.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=WANDB_PROJECT,
        help="W&B project for --resume-missing-wandb.",
    )
    parser.add_argument(
        "--wandb-timeout",
        type=int,
        default=60,
        help="W&B API timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dry_run = DRY_RUN or args.dry_run

    experiments = collect_experiments(args)
    skipped = 0

    if args.resume_missing_wandb:
        experiments, skipped = filter_missing_experiments_from_wandb(
            experiments=experiments,
            entity=args.wandb_entity,
            project=args.wandb_project,
            timeout=args.wandb_timeout,
        )

    total_experiments = len(experiments)

    print("DP-FedAug CIFAR-10 Orchestrator")
    print("=" * 50)
    print(f"Execution mode: {args.execution_mode}")
    print(f"Train sizes: {TRAIN_SIZES}")
    print(f"Partitioning strategies: {args.partitioning}")
    print(f"Alpha values (dirichlet only): {ALPHA_VALUES}")
    print(f"Target epsilon values: {TARGET_EPSILON_VALUES}")
    print(f"Synthetic counts: {SYNTHETIC_COUNTS}")
    print(f"Updates-DP enabled: {UPDATES_DP_ENABLED_VALUES}")
    print(f"Seeds: {args.starting_seed} to {args.starting_seed + args.num_seeds - 1}")
    if args.resume_missing_wandb:
        wandb_path = (
            f"{args.wandb_entity}/{args.wandb_project}"
            if args.wandb_entity else args.wandb_project
        )
        print(f"W&B resume mode: {wandb_path} ({skipped} completed runs skipped)")
    print(f"Experiments to run: {total_experiments}")
    print()

    if dry_run:
        print("DRY RUN - Listing configurations:\n")
        for idx, exp in enumerate(experiments, 1):
            eps_display = "inf (No DP)" if exp["target_epsilon"] is None else f"{exp['target_epsilon']}"
            print(
                f"[{idx}/{total_experiments}] N={exp['total_n']}, {_format_partition_info(exp)}, "
                f"e={eps_display}, Synth={exp['synthetic_count']}, "
                f"Updates-DP={_format_updates_dp_info(exp)}, Seed={exp['seed']}"
            )
        return

    if args.execution_mode == "hpc":
        submit_to_hpc(experiments)
        return

    ensure_dataset_cached()
    for idx, exp in enumerate(experiments, 1):
        print(f"\n[{idx}/{total_experiments}]")
        run_experiment_local(
            target_epsilon=exp["target_epsilon"],
            total_n=exp["total_n"],
            synthetic_count=exp["synthetic_count"],
            partitioning=exp["partitioning"],
            seed=exp["seed"],
            balancing=exp["balancing"],
            wandb_project=exp["wandb_project"],
            updates_dp_enabled=exp["updates_dp_enabled"],
            updates_dp_epsilon=exp["updates_dp_epsilon"],
            updates_dp_delta=exp["updates_dp_delta"],
            updates_dp_clipping_norm=exp["updates_dp_clipping_norm"],
            updates_dp_sensitivity=exp["updates_dp_sensitivity"],
            alpha=exp.get("alpha"),
        )


if __name__ == "__main__":
    main()
