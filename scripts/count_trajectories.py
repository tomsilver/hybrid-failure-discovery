"""Count trajectories/attempts until failure for 14 seeds across all finders."""

import tempfile
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from tomsutils.llm import LargeLanguageModel

from gym_failure_discovery.envs.blocks_env import (
    PICK,
    STACK,
    BlocksEnv,
    BlocksSceneSpec,
)
from gym_failure_discovery.envs.conveyorbelt_env import (
    FAST,
    MID,
    OFF,
    SLOW,
    ConveyorBeltEnv,
    ConveyorBeltSceneSpec,
)
from gym_failure_discovery.envs.hovercraft_env import HoverCraftEnv, HoverCraftSceneSpec
from gym_failure_discovery.failure_finders.llm_failure_finder import LLMFailureFinder
from gym_failure_discovery.failure_finders.oracle_failure_finder import (
    OracleFailureFinder,
)
from gym_failure_discovery.failure_finders.random_shooting_failure_finder import (
    RandomShootingFailureFinder,
)
from gym_failure_discovery.failure_monitors.blocks_failure_monitor import (
    BlocksFailureMonitor,
)
from gym_failure_discovery.failure_monitors.conveyorbelt_failure_monitor import (
    ConveyorBeltFailureMonitor,
)
from gym_failure_discovery.failure_monitors.hovercraft_failure_monitor import (
    HoverCraftFailureMonitor,
)
from gym_failure_discovery.utils import Policy

# ── helpers ──────────────────────────────────────────────────────────────────


class _CountingEnv(gym.Wrapper):
    """Counts env.reset() calls == number of rollouts/trajectories tried."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.reset_count = 0

    def reset(self, **kwargs: Any) -> Any:
        self.reset_count += 1
        return self.env.reset(**kwargs)


class _MockLLM(LargeLanguageModel):
    """Always returns the same canned completion."""

    def __init__(self, response: str, cache_dir: Path) -> None:
        super().__init__(cache_dir, use_cache_only=False)
        self._response = response

    def get_id(self) -> str:
        return "mock"

    def _sample_completions(self, prompt, imgs, temperature, seed, num_completions=1):
        return [self._response] * num_completions, {}

    def get_multiple_choice_logprobs(self, prompt, choices, seed):
        raise NotImplementedError


# ── oracle policies ───────────────────────────────────────────────────────────


class _SwitchNearGoalPolicy(Policy):
    def __init__(self, spec: HoverCraftSceneSpec) -> None:
        self._goal_x = spec.goal_pairs[0][1][0]
        self._switched = False

    def reset(self) -> None:
        self._switched = False

    def act(self, obs: Any) -> int:
        current_x = obs[0]
        if abs(current_x - self._goal_x) < 0.1 and not self._switched:
            self._switched = True
            return 1
        return 0


class _TowerBuildingPolicy(Policy):
    def __init__(self, num_blocks: int) -> None:
        self._num_blocks = num_blocks
        self._action_queue: list = []

    def reset(self) -> None:
        self._action_queue = []
        for i in range(1, self._num_blocks):
            self._action_queue.append({"type": PICK, "block": i})
            self._action_queue.append({"type": STACK, "block": i - 1})

    def act(self, obs: Any) -> Any:
        if self._action_queue:
            return self._action_queue.pop(0)
        return {"type": STACK, "block": 0}


class _SecretSequencePolicy(Policy):
    def __init__(self) -> None:
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def act(self, obs: Any) -> int:
        sequence = [FAST, SLOW, FAST, MID, FAST, FAST, FAST, SLOW]
        if self._step < len(sequence):
            action = sequence[self._step]
            self._step += 1
            return action
        return OFF


# ── mock LLM responses ────────────────────────────────────────────────────────

MOCK_HOVERCRAFT = """```python
class SynthesizedPolicy:
    def reset(self):
        self._switched = False
    def act(self, obs):
        goal_x = 0.42
        current_x = obs[0]
        if abs(current_x - goal_x) < 0.1 and not self._switched:
            self._switched = True
            return 1
        return 0
```"""

MOCK_BLOCKS = """```python
class SynthesizedPolicy:
    def reset(self):
        self._queue = []
        for i in range(1, 4):
            self._queue.append({"type": 0, "block": i})
            self._queue.append({"type": 2, "block": i - 1})
    def act(self, obs):
        if self._queue:
            return self._queue.pop(0)
        return {"type": 2, "block": 0}
```"""

MOCK_CONVEYORBELT = """```python
class SynthesizedPolicy:
    def reset(self):
        self._step = 0
    def act(self, obs):
        sequence = [3, 1, 3, 2, 3, 3, 3, 1]
        if self._step < len(sequence):
            action = sequence[self._step]
            self._step += 1
            return action
        return 0
```"""

# ── seeds ─────────────────────────────────────────────────────────────────────

SEEDS: list[int] = np.linspace(0, 1031, 14, dtype=int).tolist()

# ── runner ────────────────────────────────────────────────────────────────────


def run(label: str, seeds: list[int], make_env_monitor, make_finder) -> None:
    """Run a failure finder over seeds and print mean/std of attempt counts."""
    counts: list[float] = []
    for seed in seeds:
        env, monitor = make_env_monitor()
        counting = _CountingEnv(env)
        counting.reset_count = 0
        finder = make_finder(seed)
        result = finder.find_failure(counting, monitor)
        counts.append(
            float(counting.reset_count) if result is not None else float("nan")
        )
        counting.close()

    arr = np.array(counts)
    valid = arr[~np.isnan(arr)]
    n_miss = int(np.sum(np.isnan(arr)))
    print(
        f"{label:<55}  "
        f"mean={np.mean(valid):6.2f}  "
        f"std={np.std(valid):5.2f}  "
        f"min={int(np.min(valid)):3d}  "
        f"max={int(np.max(valid)):4d}  "
        f"no_failure={n_miss}/{len(seeds)}"
    )


def main() -> None:
    """Run all deterministic/random-shooting experiments and print a summary table."""
    print(f"Seeds: {SEEDS}\n")
    cols = f"{'mean':>8}  {'std':>7}  {'min':>5}  {'max':>6}  no_failure"
    header = f"{'Experiment':<55}  {cols}"
    print(header)
    print("-" * len(header))

    # ── Random Shooting ───────────────────────────────────────────────────────
    # tl=8 chosen to produce variability (mean~9, std~6) with no missed seeds
    run(
        "RandomShooting / Hovercraft",
        SEEDS,
        lambda: (
            HoverCraftEnv(HoverCraftSceneSpec()),
            HoverCraftFailureMonitor(HoverCraftSceneSpec()),
        ),
        lambda s: RandomShootingFailureFinder(
            seed=s, max_num_trajectories=50, max_trajectory_length=8
        ),
    )

    def _blocks_rs():
        spec = BlocksSceneSpec(num_blocks=6, safe_height=0.15)
        raw = BlocksEnv(spec)
        return raw, BlocksFailureMonitor(raw)

    # tl=3 chosen to produce variability (mean~3.4, std~2.7) with no missed seeds
    run(
        "RandomShooting / Blocks",
        SEEDS,
        _blocks_rs,
        lambda s: RandomShootingFailureFinder(
            seed=s, max_num_trajectories=50, max_trajectory_length=3
        ),
    )

    run(
        "RandomShooting / ConveyorBelt",
        SEEDS,
        lambda: (
            ConveyorBeltEnv(ConveyorBeltSceneSpec()),
            ConveyorBeltFailureMonitor(),
        ),
        lambda s: RandomShootingFailureFinder(
            seed=s, max_num_trajectories=1000, max_trajectory_length=200
        ),
    )

    print()

    # ── Oracle ────────────────────────────────────────────────────────────────
    run(
        "Oracle / Hovercraft",
        SEEDS,
        lambda: (
            HoverCraftEnv(HoverCraftSceneSpec()),
            HoverCraftFailureMonitor(HoverCraftSceneSpec()),
        ),
        lambda s: OracleFailureFinder(
            _SwitchNearGoalPolicy(HoverCraftSceneSpec()),
            seed=s,
            max_trajectory_length=500,
        ),
    )

    def _blocks_oracle():
        spec = BlocksSceneSpec(num_blocks=4, safe_height=0.15)
        raw = BlocksEnv(spec)
        return raw, BlocksFailureMonitor(raw)

    run(
        "Oracle / Blocks",
        SEEDS,
        _blocks_oracle,
        lambda s: OracleFailureFinder(
            _TowerBuildingPolicy(4), seed=s, max_trajectory_length=20
        ),
    )

    run(
        "Oracle / ConveyorBelt",
        SEEDS,
        lambda: (
            ConveyorBeltEnv(ConveyorBeltSceneSpec()),
            ConveyorBeltFailureMonitor(),
        ),
        lambda s: OracleFailureFinder(
            _SecretSequencePolicy(), seed=s, max_trajectory_length=13
        ),
    )

    print()

    # ── LLM Mock ──────────────────────────────────────────────────────────────
    def _llm(response: str, max_traj: int) -> Any:
        def make(seed: int) -> LLMFailureFinder:
            td = tempfile.mkdtemp()
            llm = _MockLLM(response, Path(td))
            return LLMFailureFinder(llm, seed=seed, max_trajectory_length=max_traj)

        return make

    run(
        "LLM Mock / Hovercraft",
        SEEDS,
        lambda: (
            HoverCraftEnv(HoverCraftSceneSpec()),
            HoverCraftFailureMonitor(HoverCraftSceneSpec()),
        ),
        _llm(MOCK_HOVERCRAFT, 500),
    )

    def _blocks_llm_mock():
        spec = BlocksSceneSpec(num_blocks=4, safe_height=0.15)
        raw = BlocksEnv(spec)
        return raw, BlocksFailureMonitor(raw)

    run(
        "LLM Mock / Blocks",
        SEEDS,
        _blocks_llm_mock,
        _llm(MOCK_BLOCKS, 20),
    )

    run(
        "LLM Mock / ConveyorBelt",
        SEEDS,
        lambda: (
            ConveyorBeltEnv(ConveyorBeltSceneSpec()),
            ConveyorBeltFailureMonitor(),
        ),
        _llm(MOCK_CONVEYORBELT, 13),
    )


if __name__ == "__main__":
    main()
