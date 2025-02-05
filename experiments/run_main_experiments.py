"""Run failure finding approaches on environments and save results."""

import argparse
import time
from pathlib import Path

import pandas as pd

from hybrid_failure_discovery.controllers.blocks_controller import (
    BlocksController,
)
from hybrid_failure_discovery.controllers.controller import (
    ConstraintBasedController,
)
from hybrid_failure_discovery.controllers.hovercraft_controller import (
    HoverCraftController,
)
from hybrid_failure_discovery.envs.blocks_env import (
    BlocksEnv,
)
from hybrid_failure_discovery.envs.constraint_based_env_model import (
    ConstraintBasedEnvModel,
)
from hybrid_failure_discovery.envs.hovercraft_env import HoverCraftEnv
from hybrid_failure_discovery.failure_finders.failure_finder import (
    FailureFinder,
)
from hybrid_failure_discovery.failure_finders.random_shooting_failure_finder import (
    RandomShootingFailureFinder,
)
from hybrid_failure_discovery.failure_monitors.blocks_failure_monitor import (
    BlocksFailureMonitor,
)
from hybrid_failure_discovery.failure_monitors.failure_monitor import (
    FailureMonitor,
)
from hybrid_failure_discovery.failure_monitors.hovercraft_failure_monitor import (
    HoverCraftFailureMonitor,
)


def _create_failure_finder(name: str, seed: int) -> FailureFinder:
    if name == "random-shooting":
        return RandomShootingFailureFinder(seed=seed)
    raise NotImplementedError


def _create_env(name: str, seed: int, **kwargs) -> ConstraintBasedEnvModel:
    if name == "hovercraft":
        return HoverCraftEnv(seed=seed, **kwargs)
    if name.startswith("blocks"):
        return BlocksEnv(seed=seed, **kwargs)
    raise NotImplementedError


def _create_controller(name: str, seed: int, **kwargs) -> ConstraintBasedController:
    if name == "hovercraft":
        return HoverCraftController(seed=seed, **kwargs)
    if name.startswith("blocks"):
        # This is hacky, but a way to vary the safe height for the controller.
        if "-" in name:
            _, safe_height_str = name.split("-")
            safe_height = float(safe_height_str)
        else:
            safe_height = 0.25
        return BlocksController(seed=seed, safe_height=safe_height, **kwargs)
    raise NotImplementedError


def _create_failure_monitor(name: str, **kwargs) -> FailureMonitor:
    if name == "hovercraft":
        return HoverCraftFailureMonitor(**kwargs)
    if name.startswith("blocks"):
        return BlocksFailureMonitor(**kwargs)
    raise NotImplementedError


def _main(
    failure_finder_name: str,
    domain_name: str,
    outdir: Path,
    seed: int,
    num_trials_per_seed: int,
) -> None:
    failure_finder = _create_failure_finder(failure_finder_name, seed)
    env = _create_env(domain_name, seed)

    # Should refactor this later to enforce that environments have scene specs.
    scene_spec = env.scene_spec  # type: ignore

    controller = _create_controller(domain_name, seed, scene_spec=scene_spec)
    failure_monitor = _create_failure_monitor(domain_name, scene_spec=scene_spec)

    columns = ["Failure Finder", "Domain", "Seed", "Trial", "Duration", "Success"]
    rows = []

    for trial in range(num_trials_per_seed):
        start_time = time.perf_counter()
        result = failure_finder.run(env, controller, failure_monitor)
        duration = time.perf_counter() - start_time
        success = result is not None
        row = [failure_finder_name, domain_name, seed, trial, duration, success]
        rows.append(row)

    outdir.mkdir(exist_ok=True)
    outfile = outdir / f"{failure_finder_name}__{domain_name}__{seed}.csv"
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(outfile)
    print(f"Wrote out results to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("failure_finder", type=str)
    parser.add_argument("domain", type=str)
    parser.add_argument(
        "--outdir", type=Path, default=Path(__file__).parent / "results"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    _main(args.failure_finder, args.domain, args.outdir, args.seed, args.trials)
