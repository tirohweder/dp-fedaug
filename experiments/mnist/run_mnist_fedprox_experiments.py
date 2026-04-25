"""
FedProx baseline experiment runner for MNIST.
Supports both local execution and HPC submission.
"""
import subprocess
import time
import os
import warnings
from pathlib import Path

# Suppress warnings at import time
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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

# Experiment parameters
STARTING_SEED = 301
NUM_SEEDS = 3
DRY_RUN = False  # Set to True to print configurations without running

# FedProx specific
PROXIMAL_MU_VALUES = [0.0, 0.01, 0.1, 1.0]  # mu=0 is essentially FedAvg

# Training configuration
TRAIN_SIZES = [100, 600, 1000,2000]
NUM_CLIENTS = 10
NUM_ROUNDS = 50
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
IMG_SIZE = 32

# Partitioning
# "dirichlet" uses ALPHA_VALUES for non-IID degree (lower alpha = more non-IID)
# "extreme" gives each client only 1 class (most extreme non-IID)
PARTITIONING_STRATEGIES = ["dirichlet"]
ALPHA_VALUES = [ float('inf')]  # Only used for dirichlet; inf = IID

# Logging
WANDB_PROJECT = "FedProx-MNIST-Baseline"
CLASSIFICATION_TYPE = "multiclass"

# =============================================================================
# IMPLEMENTATION
# =============================================================================


def ensure_dataset_cached():
    """Pre-download MNIST dataset if not already cached."""
    cache_dir = "data/mnist"
    if not os.path.exists(cache_dir) or not os.listdir(cache_dir):
        print("Downloading MNIST dataset (one-time only)...")
        try:
            from datasets import load_dataset
            load_dataset("mnist", cache_dir=cache_dir)
            print("MNIST dataset cached successfully.\n")
        except Exception as e:
            print(f"Could not pre-cache dataset: {e}")
            print("Will download during first run.\n")
    else:
        print("MNIST dataset already cached.\n")


def modify_pyproject_for_fedprox():
    """Modify pyproject.toml to use FedProx strategy."""
    pyproject_path = Path("pyproject.toml")
    with open(pyproject_path, 'r') as f:
        original_content = f.read()

    # Replace any strategy with fedprox
    modified_content = original_content
    for strategy in ["dpfedaug", "fedaug", "fedavg"]:
        modified_content = modified_content.replace(
            f'serverapp = "strategy.{strategy}.server_app:app"',
            'serverapp = "strategy.fedprox.server_app:app"'
        ).replace(
            f'clientapp = "strategy.{strategy}.client_app:app"',
            'clientapp = "strategy.fedprox.client_app:app"'
        )

    with open(pyproject_path, 'w') as f:
        f.write(modified_content)
    return original_content


def restore_pyproject(content):
    """Restore original pyproject.toml content."""
    with open("pyproject.toml", 'w') as f:
        f.write(content)


def get_env_vars():
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


def build_config_parts(
    total_n: int,
    partitioning: str,
    proximal_mu: float,
    seed: int,
    alpha: float = None,
    for_bash: bool = False,
):
    """Build the flwr run config parts.

    Args:
        for_bash: If True, use single quotes for strings (for bash script).
                  If False, use double quotes (for local Python subprocess).
    """
    # Use single quotes for bash scripts, double quotes for local
    q = "'" if for_bash else '"'

    config_parts = [
        f'dataset={q}mnist{q}',
        f'num-clients={NUM_CLIENTS}',
        f'img-size={IMG_SIZE}',
        f'num-server-rounds={NUM_ROUNDS}',
        f'num-local-epochs={LOCAL_EPOCHS}',
        f'lr={LEARNING_RATE}',
        f'batch-size={BATCH_SIZE}',
        f'total-n={total_n}',
        f'partitioning={q}{partitioning}{q}',
        f'wandb-project={q}{WANDB_PROJECT}{q}',
        f'seed={seed}',
        f'gradient_clipping=true',
        f'classification_type={q}{CLASSIFICATION_TYPE}{q}',
        f'weight-decay={WEIGHT_DECAY}',
        f'proximal-mu={proximal_mu}',
    ]

    # Only add alpha for dirichlet partitioning
    if partitioning == "dirichlet" and alpha is not None:
        # inf must be quoted as a string for TOML parsing
        if alpha == float('inf'):
            config_parts.insert(2, f'non-iid-alpha={q}inf{q}')
        else:
            config_parts.insert(2, f'non-iid-alpha={alpha}')

    return config_parts


def run_experiment_local(
    total_n: int,
    partitioning: str,
    proximal_mu: float,
    seed: int,
    alpha: float = None,
):
    """Run a single experiment locally."""
    original_pyproject = modify_pyproject_for_fedprox()
    env = get_env_vars()

    try:
        config_parts = build_config_parts(total_n, partitioning, proximal_mu, seed, alpha)
        cmd = ["flwr", "run", ".", "--run-config", " ".join(config_parts)]

        # Log message
        if partitioning == "extreme":
            part_info = "Part=extreme"
        else:
            alpha_str = "inf" if alpha == float('inf') else str(alpha)
            part_info = f"Part=dirichlet(a={alpha_str})"

        print(f"\n{'=' * 70}")
        print(f"FedProx-MNIST: N={total_n} | {part_info} | mu={proximal_mu} | Seed={seed}")
        print(f"{'=' * 70}")

        start_time = time.time()
        subprocess.run(cmd, check=False, env=env)
        duration = time.time() - start_time
        print(f"Finished in {duration:.1f}s")

    finally:
        restore_pyproject(original_pyproject)


def generate_lsf_script(experiments: list) -> str:
    """Generate an LSF batch script for DTU HPC."""

    # Build experiment commands
    experiment_cmds = []
    for exp in experiments:
        config_parts = build_config_parts(
            total_n=exp['total_n'],
            partitioning=exp['partitioning'],
            proximal_mu=exp['proximal_mu'],
            seed=exp['seed'],
            alpha=exp.get('alpha'),
            for_bash=True,  # Use single quotes for bash compatibility
        )
        config_str = " ".join(config_parts)
        experiment_cmds.append(f'flwr run . --run-config "{config_str}"')

    script = f'''#!/bin/bash
#BSUB -J fedprox-mnist
#BSUB -q {HPC_QUEUE}
#BSUB -W {HPC_WALLTIME}
#BSUB -n {HPC_CORES}
#BSUB -R "rusage[mem={HPC_MEM}]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num={HPC_GPUS}:mode=exclusive_process"
#BSUB -o logs/fedprox_mnist_%J.out
#BSUB -e logs/fedprox_mnist_%J.err

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

# Modify pyproject.toml for FedProx
python -c "
from pathlib import Path
p = Path('pyproject.toml')
c = p.read_text()
for s in ['dpfedaug', 'fedaug', 'fedavg']:
    c = c.replace(f'serverapp = \\"strategy.{{s}}.server_app:app\\"', 'serverapp = \\"strategy.fedprox.server_app:app\\"')
    c = c.replace(f'clientapp = \\"strategy.{{s}}.client_app:app\\"', 'clientapp = \\"strategy.fedprox.client_app:app\\"')
p.write_text(c)
"

echo "Starting FedProx MNIST experiments..."
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
            part_info = f"Part=dirichlet(a={alpha_str})"

        script += f'''
EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))
echo "========================================"
echo "[$EXPERIMENT_NUM/{len(experiment_cmds)}] N={exp['total_n']} | {part_info} | mu={exp['proximal_mu']} | Seed={exp['seed']}"
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

    # Generate the LSF script
    lsf_script = generate_lsf_script(experiments)

    # Save locally with Unix line endings
    local_script_path = Path("experiments/fedprox_mnist_hpc.sh")
    with open(local_script_path, 'w', newline='\n') as f:
        f.write(lsf_script)
    print(f"Generated LSF script: {local_script_path}")

    # Copy to HPC
    print(f"\nCopying to HPC ({HPC_HOST})...")

    try:
        # Copy script to HPC
        scp_cmd = ["scp", str(local_script_path), f"{HPC_HOST}:{HPC_PROJECT_DIR}/experiments/"]
        subprocess.run(scp_cmd, check=True)
        print("Script copied to HPC successfully!")
        print("")
        print("=" * 50)
        print("TO SUBMIT THE JOB, run these commands:")
        print("=" * 50)
        print(f"  ssh {HPC_HOST}")
        print(f"  cd {HPC_PROJECT_DIR}")
        print(f"  bsub < experiments/fedprox_mnist_hpc.sh")
        print("=" * 50)

    except subprocess.CalledProcessError as e:
        print(f"Error copying to HPC: {e}")
        print(f"LSF script saved locally at: {local_script_path}")


def collect_experiments():
    """Collect all experiment configurations."""
    experiments = []

    for total_n in TRAIN_SIZES:
        for partitioning in PARTITIONING_STRATEGIES:
            if partitioning == "extreme":
                for proximal_mu in PROXIMAL_MU_VALUES:
                    for seed_idx in range(NUM_SEEDS):
                        seed = STARTING_SEED + seed_idx
                        experiments.append({
                            'total_n': total_n,
                            'partitioning': partitioning,
                            'proximal_mu': proximal_mu,
                            'seed': seed,
                        })
            else:
                for alpha in ALPHA_VALUES:
                    for proximal_mu in PROXIMAL_MU_VALUES:
                        for seed_idx in range(NUM_SEEDS):
                            seed = STARTING_SEED + seed_idx
                            experiments.append({
                                'total_n': total_n,
                                'partitioning': partitioning,
                                'proximal_mu': proximal_mu,
                                'seed': seed,
                                'alpha': alpha,
                            })

    return experiments


def main():
    experiments = collect_experiments()
    total_experiments = len(experiments)

    print(f"FedProx MNIST Experiment Runner")
    print(f"=" * 50)
    print(f"Execution mode: {EXECUTION_MODE}")
    print(f"Total experiments: {total_experiments}")
    print(f"Train sizes (total_n): {TRAIN_SIZES}")
    print(f"Num clients: {NUM_CLIENTS}")
    print(f"Partitioning strategies: {PARTITIONING_STRATEGIES}")
    print(f"Alpha values (dirichlet only): {ALPHA_VALUES}")
    print(f"Proximal mu values: {PROXIMAL_MU_VALUES}")
    print(f"Seeds: {STARTING_SEED} to {STARTING_SEED + NUM_SEEDS - 1}")
    print()

    if DRY_RUN:
        print("DRY RUN - Listing configurations:\n")
        for idx, exp in enumerate(experiments, 1):
            if exp['partitioning'] == "extreme":
                part_info = "Part=extreme"
            else:
                alpha_str = "inf" if exp.get('alpha') == float('inf') else str(exp.get('alpha'))
                part_info = f"Part=dirichlet(a={alpha_str})"
            print(f"[{idx}/{total_experiments}] N={exp['total_n']}, {part_info}, "
                  f"mu={exp['proximal_mu']}, Seed={exp['seed']}")
        return

    if EXECUTION_MODE == "hpc":
        submit_to_hpc(experiments)
    else:
        # Local execution
        ensure_dataset_cached()

        for idx, exp in enumerate(experiments, 1):
            print(f"\n[{idx}/{total_experiments}]")
            run_experiment_local(
                total_n=exp['total_n'],
                partitioning=exp['partitioning'],
                proximal_mu=exp['proximal_mu'],
                seed=exp['seed'],
                alpha=exp.get('alpha'),
            )


if __name__ == "__main__":
    main()
