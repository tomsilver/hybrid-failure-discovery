"""Count LLM attempts until failure for 14 seeds across all three environments."""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from tomsutils.llm import OpenAILLM

from gym_failure_discovery.envs.blocks_env import BlocksEnv, BlocksSceneSpec
from gym_failure_discovery.envs.conveyorbelt_env import (
    ConveyorBeltEnv,
    ConveyorBeltSceneSpec,
)
from gym_failure_discovery.envs.hovercraft_env import HoverCraftEnv, HoverCraftSceneSpec
from gym_failure_discovery.failure_finders.llm_failure_finder import LLMFailureFinder
from gym_failure_discovery.failure_monitors.blocks_failure_monitor import (
    BlocksFailureMonitor,
)
from gym_failure_discovery.failure_monitors.conveyorbelt_failure_monitor import (
    ConveyorBeltFailureMonitor,
)
from gym_failure_discovery.failure_monitors.hovercraft_failure_monitor import (
    HoverCraftFailureMonitor,
)

SEEDS: list[int] = [0, 79, 158, 237, 317, 396, 475, 555, 634, 713, 793, 872, 951, 1031]
CACHE_DIR = Path("./llm_cache")


class _CountingEnv(gym.Wrapper):
    """Counts env.reset() calls == number of LLM policy rollouts attempted."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.reset_count = 0

    def reset(self, **kwargs: Any) -> Any:
        self.reset_count += 1
        return self.env.reset(**kwargs)


def run_env(label: str, make_env_monitor, max_traj: int, max_attempts: int) -> None:
    """Run LLM failure finder over all seeds and print mean/std of attempt counts."""
    counts: list[float] = []
    for seed in SEEDS:
        env_raw, monitor = make_env_monitor()
        counting = _CountingEnv(env_raw)
        counting.reset_count = 0
        llm = OpenAILLM("gpt-4o", CACHE_DIR, max_tokens=4096)
        finder = LLMFailureFinder(
            llm,
            seed=seed,
            max_trajectory_length=max_traj,
            max_num_attempts=max_attempts,
        )
        result = finder.find_failure(counting, monitor)
        n = float(counting.reset_count) if result is not None else float("nan")
        status = f"found in {int(n)} attempt(s)" if result is not None else "NOT FOUND"
        print(f"  seed={seed:4d}  {status}")
        counts.append(n)
        counting.close()

    arr = np.array(counts)
    valid = arr[~np.isnan(arr)]
    n_miss = int(np.sum(np.isnan(arr)))
    print(
        f"\n  {label}:\n"
        f"    mean={np.mean(valid):.2f}  std={np.std(valid):.2f}  "
        f"min={int(np.min(valid))}  max={int(np.max(valid))}  "
        f"no_failure={n_miss}/{len(SEEDS)}\n"
    )


def main() -> None:
    """Run real LLM failure finder experiments and report statistics."""
    print(f"Seeds: {SEEDS}\n")

    # ── Hovercraft ────────────────────────────────────────────────────────────
    print("=== LLM Real / Hovercraft (max_attempts=5, max_traj_length=500) ===")
    run_env(
        "LLM Real / Hovercraft",
        lambda: (
            HoverCraftEnv(HoverCraftSceneSpec()),
            HoverCraftFailureMonitor(HoverCraftSceneSpec()),
        ),
        max_traj=500,
        max_attempts=5,
    )

    # ── Blocks ────────────────────────────────────────────────────────────────
    print("=== LLM Real / Blocks (max_attempts=5, max_traj_length=20) ===")

    def _blocks() -> tuple:
        spec = BlocksSceneSpec(num_blocks=4, safe_height=0.15)
        raw = BlocksEnv(spec)
        return raw, BlocksFailureMonitor(raw)

    run_env("LLM Real / Blocks", _blocks, max_traj=20, max_attempts=5)

    # ── ConveyorBelt ──────────────────────────────────────────────────────────
    print("=== LLM Real / ConveyorBelt (max_attempts=10, max_traj_length=200) ===")
    run_env(
        "LLM Real / ConveyorBelt",
        lambda: (
            ConveyorBeltEnv(ConveyorBeltSceneSpec(), render_mode="rgb_array"),
            ConveyorBeltFailureMonitor(),
        ),
        max_traj=200,
        max_attempts=10,
    )


if __name__ == "__main__":
    main()
