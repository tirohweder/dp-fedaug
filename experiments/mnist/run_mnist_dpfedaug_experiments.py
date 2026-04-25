import argparse
import subprocess
import time
import os
import warnings
from pathlib import Path

# Suppress warnings at import time
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#
# =============================================================================
# CONFIGURATION - Edit these values directly (no command line args)
# =============================================================================

# Execution mode: "local" or "hpc"
EXECUTION_MODE = "hpc"

# HPC Configuration (only used when EXECUTION_MODE = "hpc").
# Override via env vars in your .env file (see .env.example).
HPC_HOST = os.environ.get("HPC_HOST", "user@hpc.example.com")
HPC_PROJECT_DIR = os.environ.get("HPC_PROJECT_DIR", "~/dp-fedaug")
HPC_QUEUE = "gpuv100"  # LSF queue (gpuv100, gpua100, hpc)
HPC_WALLTIME = "24:00"  # Job time limit (HH:MM)
HPC_GPUS = 1  # Number of GPUs (use "num=1:mode=exclusive_process")
HPC_CORES = 4  # Number of CPU cores
HPC_MEM = "32GB"  # Memory per core




# =============================================================================
# EXPERIMENT GRID — N=10000 isolation study: baseline / DP-FedAug / full pipeline
# =============================================================================
# Conditions:
#   1. Baseline:      synth=0,  updates_dp=False
#   2. DP-FedAug:     synth=50, updates_dp=False, target_epsilon=8 (VAE DP only)
#   3. Full pipeline: synth=50, updates_dp=True,  target_epsilon=8, updates_epsilon=8
# Partitioning: extreme + dirichlet α=0.1 (most challenging non-IID regimes)
# Seeds: 3 for statistical validity
# =============================================================================
SYNTHETIC_COUNTS = [0]        # 0=baseline FedAvg, 50=augmented
TARGET_EPSILON_VALUES = [0]       # ε=8 for VAE generation
ALPHA_VALUES = [0.1]              # strong non-IID (extreme uses its own partitioning)
TRAIN_SIZES = [10000]             # 1000 samples/client → sample_rate=0.032, low noise

# DP on model updates — uses Opacus DP-SGD (per-sample gradient noise), NOT LocalDpMod.
# Privacy guarantee: (ε, δ)-local DP covering the ENTIRE training (all FL rounds combined).
# Noise multiplier is computed upfront from (ε, δ, total_steps, sample_rate) so the stated
# ε is the true total budget, not a per-round budget.
# max_grad_norm: per-sample gradient clipping (Opacus default 1.0 works well for CNNs).
UPDATES_DP_ENABLED_VALUES = [True]  # both on and off (filtered by should_skip)
UPDATES_DP_EPSILON_VALUES = [8]            # match TARGET_EPSILON_VALUES above
UPDATES_DP_DELTA_VALUES = [1e-5]
UPDATES_DP_MAX_GRAD_NORMS = [1.0]  # per-sample gradient clipping norm for Opacus DP-SGD

STARTING_SEED = 301
NUM_SEEDS = 3     # 3 seeds for statistical validity
NUM_CLIENTS = 10
NUM_ROUNDS_VALUES = [50]
LOCAL_EPOCHS_VALUES = [1]
SYNTHETIC_EPOCHS = 100
SYNTHETIC_BATCH_SIZE = 64
SYNTHETIC_LATENT_DIM = 20
SYNTHETIC_KL_WARMUP = 10
SYNTHETIC_LR = 0.001
SYNTHETIC_DELTA = 1e-5
MAX_GRAD_NORM = 1.0
IMG_SIZE = 32
BATCH_SIZE = 32
LEARNING_RATE = 0.01


CLASSIFICATION_TYPE = "multiclass"  # "binary" or "multiclass"
BALANCING = "scaled"  # "none" or "scaled"
MNIST_USE_DROPOUT = True
MNIST_DROPOUT_RATE = 0.1

# Partitioning strategies:
# - "dirichlet": Standard Dirichlet-based non-IID partitioning (uses alpha)
# - "extreme": Each client gets data from only ONE label (for 10 clients/10 classes)
#              This creates maximum heterogeneity to test synthetic data augmentation
PARTITIONING_STRATEGIES = ["dirichlet", "extreme"]

WANDB_PROJECT = "DP-FedAug-MNIST-UpdatesDP-Study"


def ensure_dataset_cached():
    """Pre-download MNIST dataset if not already cached."""
    cache_dir = "data/mnist"
    if not os.path.exists(cache_dir) or not os.listdir(cache_dir):
        print("📥 Downloading MNIST dataset (one-time only)...")
        try:
            from datasets import load_dataset
            load_dataset("mnist", cache_dir=cache_dir)
            print("✅ MNIST dataset cached successfully.\n")
        except Exception as e:
            print(f"⚠️ Could not pre-cache dataset: {e}")
            print("   Will download during first run.\n")
    else:
        print("✅ MNIST dataset already cached.\n")


def get_env_vars():
    """Get environment variables for subprocess."""
    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env["GRPC_VERBOSITY"] = "ERROR"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["HF_HUB_VERBOSITY"] = "error"
    env["DATASETS_VERBOSITY"] = "error"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_SILENT"] = "true"
    env["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::FutureWarning,ignore::UserWarning"
    env["RAY_DEDUP_LOGS"] = "1"
    env["TF_ENABLE_ONEDNN_OPTS"] = "0"
    env["KERAS_BACKEND"] = "torch"
    return env


def modify_pyproject_for_dpfedaug():
    pyproject_path = Path("pyproject.toml")
    with open(pyproject_path, 'r') as f:
        original_content = f.read()
    
    modified_content = original_content.replace(
        'serverapp = "strategy.fedaug.server_app:app"',
        'serverapp = "strategy.dpfedaug.server_app:app"'
    ).replace(
        'clientapp = "strategy.fedaug.client_app:app"',
        'clientapp = "strategy.dpfedaug.client_app:app"'
    )
    
    with open(pyproject_path, 'w') as f:
        f.write(modified_content)
    return original_content

def restore_pyproject(content):
    with open("pyproject.toml", 'w') as f:
        f.write(content)


def build_config_parts(
    target_epsilon: float,
    total_n: int,
    synthetic_count: int,
    partitioning: str,
    seed: int,
    balancing: str,
    updates_dp_enabled: bool,
    num_rounds: int,
    local_epochs: int,
    updates_dp_epsilon: float = None,
    updates_dp_delta: float = None,
    updates_dp_max_grad_norm: float = None,
    alpha: float = None,
    for_bash: bool = False,
):
    """Build the flwr run config parts.

    Args:
        for_bash: If True, use single quotes for strings (for bash script).
                  If False, use double quotes (for local Python subprocess).
    """
    q = "'" if for_bash else '"'
    target_eps_str = "none" if target_epsilon is None else str(target_epsilon)

    def format_value(value):
        if isinstance(value, str):
            return f"{q}{value}{q}"
        return str(value)

    config_parts = [
        f"dataset={q}mnist{q}",
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
        f"img-size={IMG_SIZE}",
        f"num-server-rounds={num_rounds}",
        f"num-local-epochs={local_epochs}",
        f"lr={LEARNING_RATE}",
        f"batch-size={BATCH_SIZE}",
        f"total-n={total_n}",
        f"partitioning={q}{partitioning}{q}",
        f"balancing={q}{balancing}{q}",
        f"wandb-project={q}{WANDB_PROJECT}{q}",
        f"seed={seed}",
        f"gradient_clipping=true",
        f"classification_type={q}{CLASSIFICATION_TYPE}{q}",
        f"mnist-use-dropout={str(MNIST_USE_DROPOUT).lower()}",
        f"mnist-dropout-rate={MNIST_DROPOUT_RATE}",
        f"updates-dp-enabled={str(updates_dp_enabled).lower()}",
        f"updates-dp-epsilon={format_value(updates_dp_epsilon if updates_dp_enabled else 'none')}",
        f"updates-dp-delta={format_value(updates_dp_delta if updates_dp_enabled else 'none')}",
        f"updates-dp-max-grad-norm={format_value(updates_dp_max_grad_norm if updates_dp_enabled else 'none')}",
    ]

    # Log non-iid-alpha for both partitioning modes
    if partitioning == "dirichlet" and alpha is not None:
        alpha_str = "inf" if alpha == float('inf') else str(alpha)
        if alpha == float('inf'):
            config_parts.insert(2, f"non-iid-alpha={q}{alpha_str}{q}")
        else:
            config_parts.insert(2, f"non-iid-alpha={alpha_str}")
    elif partitioning == "extreme":
        config_parts.insert(2, f"non-iid-alpha={q}extreme{q}")

    return config_parts

def run_experiment(
    target_epsilon: float,
    total_n: int,
    synthetic_count: int,
    partitioning: str,
    seed: int,
    balancing: str,
    updates_dp_enabled: bool,
    num_rounds: int,
    local_epochs: int,
    updates_dp_epsilon: float = None,
    updates_dp_delta: float = None,
    updates_dp_max_grad_norm: float = None,
    alpha: float = None
):
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
            updates_dp_enabled=updates_dp_enabled,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            updates_dp_epsilon=updates_dp_epsilon,
            updates_dp_delta=updates_dp_delta,
            updates_dp_max_grad_norm=updates_dp_max_grad_norm,
            alpha=alpha,
        )

        cmd = ["flwr", "run", ".", "--run-config", " ".join(config_parts)]

        # Build informative log message
        if partitioning == "extreme":
            part_info = "Part=extreme"
        else:
            alpha_str = "inf" if alpha == float('inf') else str(alpha)
            part_info = f"Part=dirichlet(α={alpha_str})"

        eps_display = "∞ (No DP)" if target_epsilon is None else f"{target_epsilon}"

        updates_dp_display = "off"
        if updates_dp_enabled:
            updates_dp_display = (
                f"on (ε={updates_dp_epsilon}, δ={updates_dp_delta}, "
                f"max_grad_norm={updates_dp_max_grad_norm})"
            )
        print(f"\n{'=' * 70}")
        print(f"DP-MNIST: N={total_n} | {part_info} | ε={eps_display} | "
              f"Synth={synthetic_count} | Rounds={num_rounds} | Epochs={local_epochs} | "
              f"Updates-DP={updates_dp_display} | Seed={seed}")
        print(f"{'=' * 70}")

        start_time = time.time()
        subprocess.run(cmd, check=False, env=env)
        duration = time.time() - start_time
        print(f"Finished in {duration:.1f}s")
        
    finally:
        restore_pyproject(original_pyproject)


def generate_lsf_script(experiments: list) -> str:
    """Generate an LSF batch script for DTU HPC."""
    experiment_cmds = []
    for exp in experiments:
        config_parts = build_config_parts(
            target_epsilon=exp['target_epsilon'],
            total_n=exp['total_n'],
            synthetic_count=exp['synthetic_count'],
            partitioning=exp['partitioning'],
            seed=exp['seed'],
            balancing=exp['balancing'],
            updates_dp_enabled=exp['updates_dp_enabled'],
            num_rounds=exp['num_rounds'],
            local_epochs=exp['local_epochs'],
            updates_dp_epsilon=exp.get('updates_dp_epsilon'),
            updates_dp_delta=exp.get('updates_dp_delta'),
            updates_dp_max_grad_norm=exp.get('updates_dp_max_grad_norm'),
            alpha=exp.get('alpha'),
            for_bash=True,
        )
        config_str = " ".join(config_parts)
        experiment_cmds.append(f'flwr run . --run-config "{config_str}"')

    script = f'''#!/bin/bash
#BSUB -J dpfedaug-mnist
#BSUB -q {HPC_QUEUE}
#BSUB -W {HPC_WALLTIME}
#BSUB -n {HPC_CORES}
#BSUB -R "rusage[mem={HPC_MEM}]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num={HPC_GPUS}:mode=exclusive_process"
#BSUB -o logs/dpfedaug_mnist_%J.out
#BSUB -e logs/dpfedaug_mnist_%J.err

# Load modules
module load python3/3.12.4
module load cuda/12.1

# Change to project directory
cd {HPC_PROJECT_DIR}

# Create logs directory
mkdir -p logs

# Activate virtual environment (REQUIRED)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Activated venv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Activated .venv/bin/activate"
else
    echo "ERROR: No virtual environment found!"
    echo "Please create one with: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Verify flwr is installed
if ! command -v flwr &> /dev/null; then
    echo "ERROR: flwr command not found!"
    echo "Please install with: pip install flwr"
    exit 1
fi
echo "flwr version: $(flwr --version)"

# Set environment variables
export TF_CPP_MIN_LOG_LEVEL=3
export GRPC_VERBOSITY=ERROR
export WANDB_SILENT=true
export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::FutureWarning,ignore::UserWarning"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_VERBOSITY=error
export DATASETS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=1
export TF_ENABLE_ONEDNN_OPTS=0
export KERAS_BACKEND=torch

# Load .env for WANDB_API_KEY (and any other vars)
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
    echo "Loaded .env"
else
    echo "WARN: .env not found"
fi

# Default W&B mode based on API key availability
if [ -n "$WANDB_API_KEY" ]; then
    export WANDB_MODE=online
else
    export WANDB_MODE=offline
fi

# Ensure dataset is cached
python -c "from datasets import load_dataset; load_dataset('mnist', cache_dir='data/mnist')" 2>/dev/null || true

# Modify pyproject.toml for DPFedAug
python -c "
from pathlib import Path
p = Path('pyproject.toml')
c = p.read_text()
for s in ['fedaug', 'fedavg', 'fedprox']:
    c = c.replace(f'serverapp = \\"strategy.{{s}}.server_app:app\\"', 'serverapp = \\"strategy.dpfedaug.server_app:app\\"')
    c = c.replace(f'clientapp = \\"strategy.{{s}}.client_app:app\\"', 'clientapp = \\"strategy.dpfedaug.client_app:app\\"')
p.write_text(c)
"

echo "Starting DPFedAug MNIST experiments..."
echo "Total experiments: {len(experiment_cmds)}"
echo ""

# Run experiments
EXPERIMENT_NUM=0
'''

    for i, cmd in enumerate(experiment_cmds):
        exp = experiments[i]
        if exp['partitioning'] == "extreme":
            part_info = "Part=extreme"
        else:
            alpha_str = "inf" if exp.get('alpha') == float('inf') else str(exp.get('alpha'))
            part_info = f"Part=dirichlet(α={alpha_str})"

        eps_display = "∞ (No DP)" if exp['target_epsilon'] is None else f"{exp['target_epsilon']}"
        updates_dp_display = "off" if not exp['updates_dp_enabled'] else (
            f"on (ε={exp['updates_dp_epsilon']}, δ={exp['updates_dp_delta']}, "
            f"max_grad_norm={exp['updates_dp_max_grad_norm']})"
        )

        script += f'''
EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))
echo "========================================"
echo "[$EXPERIMENT_NUM/{len(experiment_cmds)}] N={exp['total_n']} | {part_info} | ε={eps_display} | Synth={exp['synthetic_count']} | Rounds={exp['num_rounds']} | Epochs={exp['local_epochs']} | Seed={exp['seed']}"
echo "========================================"
{cmd}
'''

    script += '''
echo ""
echo "All experiments completed!"
'''
    return script


def submit_to_hpc(experiments: list):
    """Generate LSF script and copy to DTU HPC."""
    lsf_script = generate_lsf_script(experiments)

    local_script_path = Path("hpc/dpfedaug_mnist_hpc.sh")
    with open(local_script_path, 'w', newline='\n') as f:
        f.write(lsf_script)
    print(f"Generated LSF script: {local_script_path}")

    print(f"\nCopying to HPC ({HPC_HOST})...")
    try:
        scp_cmd = ["scp", str(local_script_path), f"{HPC_HOST}:{HPC_PROJECT_DIR}/experiments/"]
        subprocess.run(scp_cmd, check=True)
        print("Script copied to HPC successfully!")
        print("")
        print("=" * 50)
        print("TO SUBMIT THE JOB, run these commands:")
        print("=" * 50)
        print(f"  ssh {HPC_HOST}")
        print(f"  cd {HPC_PROJECT_DIR}")
        print(f"  bsub < hpc/dpfedaug_mnist_hpc.sh")
        print("=" * 50)
    except subprocess.CalledProcessError as e:
        print(f"Error copying to HPC: {e}")
        print(f"LSF script saved locally at: {local_script_path}")

def main():
    parser = argparse.ArgumentParser(description="Run MNIST DP-FedAug Experiments")
    parser.add_argument("--num-seeds", type=int, default=NUM_SEEDS, help="Number of seeds to run per configuration")
    parser.add_argument("--starting-seed", type=int, default=STARTING_SEED, help="Starting seed value")
    parser.add_argument("--dry-run", action="store_true", help="Print configurations without running")
    parser.add_argument(
        "--partitioning",
        type=str,
        nargs="+",
        default=PARTITIONING_STRATEGIES,
        choices=["dirichlet", "extreme"],
        help="Partitioning strategies to test (default: dirichlet)"
    )
    parser.add_argument(
        "--balancing",
        type=str,
        default=BALANCING,
        choices=["none", "scaled"],
        help="Synthetic balancing strategy (default: none)"
    )
    args = parser.parse_args()

    def iter_updates_dp_configs():
        for enabled in UPDATES_DP_ENABLED_VALUES:
            if not enabled:
                yield False, None, None, None
            else:
                for epsilon in UPDATES_DP_EPSILON_VALUES:
                    for delta in UPDATES_DP_DELTA_VALUES:
                        for max_grad_norm in UPDATES_DP_MAX_GRAD_NORMS:
                            yield True, epsilon, delta, max_grad_norm

    def should_skip(synthetic_count, target_epsilon, updates_dp_enabled, updates_dp_epsilon):
        """Filter invalid experiment combinations."""
        # Only run matching epsilon pairs (ε_syn == ε_update) to avoid explosion
        if updates_dp_enabled and target_epsilon is not None and updates_dp_epsilon != target_epsilon:
            return True
        return False

    experiments = []
    for total_n in TRAIN_SIZES:
      for num_rounds in NUM_ROUNDS_VALUES:
        for local_epochs in LOCAL_EPOCHS_VALUES:
            for partitioning in args.partitioning:
                if partitioning == "extreme":
                    for synthetic_count in SYNTHETIC_COUNTS:
                        eps_values = [None] if synthetic_count == 0 else TARGET_EPSILON_VALUES
                        for target_epsilon in eps_values:
                            for updates_dp_enabled, updates_dp_epsilon, updates_dp_delta, updates_dp_max_grad_norm in iter_updates_dp_configs():
                                if should_skip(synthetic_count, target_epsilon, updates_dp_enabled, updates_dp_epsilon):
                                    continue
                                for seed_idx in range(args.num_seeds):
                                    seed = args.starting_seed + seed_idx
                                    experiments.append({
                                        "target_epsilon": target_epsilon,
                                        "total_n": total_n,
                                        "synthetic_count": synthetic_count,
                                        "partitioning": partitioning,
                                        "seed": seed,
                                        "balancing": args.balancing,
                                        "num_rounds": num_rounds,
                                        "local_epochs": local_epochs,
                                        "updates_dp_enabled": updates_dp_enabled,
                                        "updates_dp_epsilon": updates_dp_epsilon,
                                        "updates_dp_delta": updates_dp_delta,
                                        "updates_dp_max_grad_norm": updates_dp_max_grad_norm,
                                    })
                else:
                    for alpha in ALPHA_VALUES:
                        for synthetic_count in SYNTHETIC_COUNTS:
                            eps_values = [None] if synthetic_count == 0 else TARGET_EPSILON_VALUES
                            for target_epsilon in eps_values:
                                for updates_dp_enabled, updates_dp_epsilon, updates_dp_delta, updates_dp_max_grad_norm in iter_updates_dp_configs():
                                    if should_skip(synthetic_count, target_epsilon, updates_dp_enabled, updates_dp_epsilon):
                                        continue
                                    for seed_idx in range(args.num_seeds):
                                        seed = args.starting_seed + seed_idx
                                        experiments.append({
                                            "target_epsilon": target_epsilon,
                                            "total_n": total_n,
                                            "synthetic_count": synthetic_count,
                                            "partitioning": partitioning,
                                            "seed": seed,
                                            "balancing": args.balancing,
                                            "num_rounds": num_rounds,
                                            "local_epochs": local_epochs,
                                            "updates_dp_enabled": updates_dp_enabled,
                                            "updates_dp_epsilon": updates_dp_epsilon,
                                            "updates_dp_delta": updates_dp_delta,
                                            "updates_dp_max_grad_norm": updates_dp_max_grad_norm,
                                            "alpha": alpha,
                                        })

    total_experiments = len(experiments)

    print(f"DP-FedAug MNIST Orchestrator: {total_experiments} experiments to run")
    print(f"Execution mode: {EXECUTION_MODE}")
    print(f"Partitioning strategies: {args.partitioning}")
    print(f"Train sizes: {TRAIN_SIZES}")
    print(f"Num rounds: {NUM_ROUNDS_VALUES}")
    print(f"Local epochs: {LOCAL_EPOCHS_VALUES}")
    print(f"Target epsilon values: {TARGET_EPSILON_VALUES}")
    print(f"Synthetic counts: {SYNTHETIC_COUNTS}")
    print()

    if args.dry_run:
        print("DRY RUN - Listing configurations:\n")
        for idx, exp in enumerate(experiments, 1):
            if exp['partitioning'] == "extreme":
                part_info = "Part=extreme"
            else:
                alpha_str = "inf" if exp.get('alpha') == float('inf') else str(exp.get('alpha'))
                part_info = f"Part=dirichlet(α={alpha_str})"

            eps_display = "∞ (No DP)" if exp['target_epsilon'] is None else f"{exp['target_epsilon']}"
            updates_dp_display = "off" if not exp['updates_dp_enabled'] else (
                f"on (ε={exp['updates_dp_epsilon']}, δ={exp['updates_dp_delta']}, "
                f"max_grad_norm={exp['updates_dp_max_grad_norm']})"
            )
            print(f"[{idx}/{total_experiments}] N={exp['total_n']}, {part_info}, "
                  f"ε={eps_display}, Synth={exp['synthetic_count']}, Rounds={exp['num_rounds']}, "
                  f"Epochs={exp['local_epochs']}, Updates-DP={updates_dp_display}, Seed={exp['seed']}")
        return

    if EXECUTION_MODE == "hpc":
        submit_to_hpc(experiments)
        return

    # Local execution
    ensure_dataset_cached()
    for idx, exp in enumerate(experiments, 1):
        print(f"\n[{idx}/{total_experiments}]")
        run_experiment(
            target_epsilon=exp['target_epsilon'],
            total_n=exp['total_n'],
            synthetic_count=exp['synthetic_count'],
            partitioning=exp['partitioning'],
            seed=exp['seed'],
            balancing=exp['balancing'],
            updates_dp_enabled=exp['updates_dp_enabled'],
            num_rounds=exp['num_rounds'],
            local_epochs=exp['local_epochs'],
            updates_dp_epsilon=exp['updates_dp_epsilon'],
            updates_dp_delta=exp['updates_dp_delta'],
            updates_dp_max_grad_norm=exp['updates_dp_max_grad_norm'],
            alpha=exp.get('alpha'),
        )


if __name__ == "__main__":
    main()
