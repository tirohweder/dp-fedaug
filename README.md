# DP-FedAug: Differentially Private Federated Augmentation

The codebase accompanying the thesis *"(Provably) Private Federated Augmentation
Addressing Small-Sample, Non-IID Challenges via Differentially Private Modeling"*

## Overview

Cross-silo federated learning enables institutions to train shared models without centralizing
sensitive data, but performance often degrades when local datasets are small and non-IID. 
This thesis proposes Differentially Private Federated Augmentation (DP-FedAug), a
two-phase framework in which clients first train differentially private generative models to
produce synthetic data with item-level (ω, ε)-differential privacy guarantees, and then use
redistributed synthetic samples to augment local datasets for federated training. A full
pipeline additionally applies DP-SGD to communicated model updates, providing formal
privacy guarantees for both released synthetic data and transmitted updates.
Experiments show that DP-FedAug is most elective in highly heterogeneous, data-scarce

settings. On MNIST, with N_budget = 600 real samples distributed across K = 10 clients
and Dirichlet heterogeneity ϑ = 0.1, accuracy improves from 80.3% without augmentation
to 87.1% at ω = 1, compared with a non-private upper bound of 92.7%. Under pathological
one-class-per-client partitions, augmentation remains beneficial, though strict privacy limits
recovery more strongly. The full pipeline outperforms FedAvg with update-level DP alone,
especially under severe heterogeneity. Overall, the results show that private synthetic
augmentation can mitigate client drift, but its usefulness depends strongly on the privacy
budget, data complexity, and heterogeneity structure.

The codebase implements the strategy on top of [Flower](https://flower.ai/)
with [Opacus](https://opacus.ai/) for DP.



## Setup

The project uses [`uv`](https://github.com/astral-sh/uv) for dependency
management.

```bash
# 1. Install uv (see https://github.com/astral-sh/uv#installation)
# 2. Sync the environment
uv sync

# 3. Copy the env template and fill in your own values
cp .env.example .env
# Edit .env to set WANDB_API_KEY, HUGGING_KEY, and (optionally) HPC_*
```

Required env vars (see `.env.example`):

| Variable        | Purpose                                        | Required? |
| --------------- | ---------------------------------------------- | --------- |
| `WANDB_API_KEY` | Weights & Biases experiment logging            | yes       |
| `HUGGING_KEY`   | Hugging Face downloads (some pretrained nets)  | optional  |
| `HPC_*`         | Remote Slurm/LSF submission (else run local)   | optional  |

## Datasets

Datasets are downloaded into `data/` (gitignored). MNIST and CIFAR-10 are
fetched automatically by their loaders; the brain tumor pipeline requires
manual download (see the corresponding `data_loader/` and `notebooks/`
subdirectories for instructions).

## Running experiments

### Local quick start (MNIST DP-FedAug)

```bash
# Edit experiments/mnist/run_mnist_dpfedaug_experiments.py to set
# EXECUTION_MODE = "local" and adjust the experiment grid as desired.
uv run python experiments/mnist/run_mnist_dpfedaug_experiments.py
```

The Flower simulation runs locally with the configuration in
`pyproject.toml > [tool.flwr.app.config]`. Results are logged to W&B.

### HPC submission

If you have access to a LSF cluster, set the `HPC_*` variables in `.env`
and switch `EXECUTION_MODE = "hpc"` in the runner script. The runners use the
templates in `hpc/` to generate and submit jobs.

## Reproducing thesis figures

Once experiment runs are logged to W&B, the integrated exporter in `visual/`
pulls results back and renders the thesis plots and study outputs:

```bash
# List available export targets
uv run python -m visual.report --list

# Export the numbered thesis figures
uv run python -m visual.report thesis

# Export a specific study
uv run python -m visual.report mnist-seeded

# Export everything
uv run python -m visual.report all
```

Figures are written to `visual/outputs/thesis/` (numbered figures) and
`visual/outputs/<study>/` (per-study exports). Set
`THESIS_PICTURES_DIR=/path/to/external/picture/dir` in `.env` to additionally
mirror the numbered figures to an external directory.

## Citation

If you use this codebase or its results, please cite the thesis:

```bibtex
@mastersthesis{rohweder2026dpfedaug,
  author = {Rohweder, Timm},
  title  = {(Provably) Private Federated Augmentation
Addressing Small-Sample, Non-IID Challenges via Differentially Private Modeling},
  school = {Technical University of Denmark (DTU)},
  year   = {2026}
}
```


## Acknowledgements

Built on top of [Flower](https://flower.ai/),
[Opacus](https://opacus.ai/), and
[Weights & Biases](https://wandb.ai/). Parts of the implementation build on
earlier exploratory versions developed for this project.
