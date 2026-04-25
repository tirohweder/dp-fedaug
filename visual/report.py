from __future__ import annotations

import argparse
from dataclasses import dataclass
import inspect
from typing import Callable

from visual.export_all import export_all_figures
from visual.study_cifar_seeded import export_study as export_cifar_seeded
from visual.study_mnist_dpfedaug import export_study as export_mnist_dpfedaug
from visual.study_mnist_exploration import export_study as export_mnist_exploration
from visual.study_mnist_fedprox import export_study as export_mnist_fedprox
from visual.study_mnist_seeded import export_study as export_mnist_seeded


@dataclass(frozen=True)
class ExportTarget:
    name: str
    description: str
    exporter: Callable[..., object]


TARGETS = {
    "thesis": ExportTarget(
        name="thesis",
        description="Export the numbered thesis figures and appendix heatmaps.",
        exporter=export_all_figures,
    ),
    "mnist-exploration": ExportTarget(
        name="mnist-exploration",
        description="Export the MNIST DP exploration study.",
        exporter=export_mnist_exploration,
    ),
    "mnist-dpfedaug": ExportTarget(
        name="mnist-dpfedaug",
        description="Export the full MNIST DP-FedAug study.",
        exporter=export_mnist_dpfedaug,
    ),
    "mnist-seeded": ExportTarget(
        name="mnist-seeded",
        description="Export the seeded MNIST DP-FedAug study.",
        exporter=export_mnist_seeded,
    ),
    "cifar-seeded": ExportTarget(
        name="cifar-seeded",
        description="Export the seeded CIFAR-10 DP-FedAug study.",
        exporter=export_cifar_seeded,
    ),
    "mnist-fedprox": ExportTarget(
        name="mnist-fedprox",
        description="Export the MNIST FedProx baseline study.",
        exporter=export_mnist_fedprox,
    ),
}


def run_targets(target_names: list[str], *, refresh: bool = False, entity: str | None = None, timeout: int | None = None):
    selected = list(TARGETS) if "all" in target_names else target_names
    seen = set()
    for name in selected:
        if name in seen:
            continue
        seen.add(name)
        target = TARGETS[name]
        supported = inspect.signature(target.exporter).parameters
        kwargs = {}
        if "refresh" in supported:
            kwargs["refresh"] = refresh
        if "entity" in supported:
            kwargs["entity"] = entity
        if timeout is not None and "timeout" in supported:
            kwargs["timeout"] = timeout
        print(f"\n=== {target.name} ===")
        target.exporter(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Run integrated visual exports.")
    parser.add_argument(
        "targets",
        nargs="*",
        default=["thesis"],
        help="Export target(s): thesis, mnist-exploration, mnist-dpfedaug, mnist-seeded, cifar-seeded, mnist-fedprox, or all.",
    )
    parser.add_argument("--refresh", action="store_true", help="Re-fetch data from W&B instead of using cached CSV files.")
    parser.add_argument("--entity", type=str, default=None, help="Optional W&B entity/org.")
    parser.add_argument("--timeout", type=int, default=None, help="Optional W&B timeout override in seconds.")
    parser.add_argument("--list", action="store_true", help="List available export targets and exit.")
    args = parser.parse_args()

    if args.list:
        for target in TARGETS.values():
            print(f"{target.name:<18} {target.description}")
        print(f"{'all':<18} Run every registered export target.")
        return

    invalid = [name for name in args.targets if name not in TARGETS and name != "all"]
    if invalid:
        raise SystemExit(f"Unknown export target(s): {', '.join(invalid)}")

    run_targets(args.targets, refresh=args.refresh, entity=args.entity, timeout=args.timeout)


if __name__ == "__main__":
    main()
