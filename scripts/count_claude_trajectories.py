"""Count trajectories until failure for the Claude-generated approaches.

Runs each GeneratedApproach (blocks, conveyorbelt, hovercraft) across 14
seeds and reports mean, std, min, max, and no_failure counts — matching
the format of count_trajectories.py.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from tomsutils.motion_planning import BiRRT

from gym_failure_discovery.envs.blocks_env import BlocksEnv, BlocksSceneSpec
from gym_failure_discovery.envs.conveyorbelt_env import (
    ConveyorBeltEnv,
    ConveyorBeltSceneSpec,
)
from gym_failure_discovery.envs.hovercraft_env import HoverCraftEnv, HoverCraftSceneSpec
from gym_failure_discovery.failure_monitor_wrapper import FailureMonitorWrapper
from gym_failure_discovery.failure_monitors.blocks_failure_monitor import (
    BlocksFailureMonitor,
)
from gym_failure_discovery.failure_monitors.conveyorbelt_failure_monitor import (
    ConveyorBeltFailureMonitor,
)
from gym_failure_discovery.failure_monitors.hovercraft_failure_monitor import (
    HoverCraftFailureMonitor,
)
from gym_failure_discovery.failure_monitors.failure_monitor import FailureMonitor

# ── Minimal CSP stubs ────────────────────────────────────────────────────────
# The GeneratedApproach classes expect a `primitives['csp']` module with
# CSPVariable, FunctionalCSPConstraint, CSP, FunctionalCSPSampler, and
# RandomWalkCSPSolver.  We provide lightweight implementations that are
# sufficient for the approaches to run.


class CSPVariable:
    """A variable in a CSP problem."""

    def __init__(self, name: str, space: gym.spaces.Space) -> None:
        self.name = name
        self.space = space

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CSPVariable):
            return NotImplemented
        return self.name == other.name

    def __repr__(self) -> str:
        return f"CSPVariable({self.name!r})"


class FunctionalCSPConstraint:
    """A constraint defined by a callable."""

    def __init__(
        self, name: str, variables: list[CSPVariable], fn: Any
    ) -> None:
        self.name = name
        self.variables = variables
        self.fn = fn

    def check(self, assignment: dict[CSPVariable, Any]) -> bool:
        args = [assignment[v] for v in self.variables]
        return bool(self.fn(*args))


class CSP:
    """A constraint satisfaction problem."""

    def __init__(
        self,
        variables: list[CSPVariable],
        constraints: list[FunctionalCSPConstraint],
    ) -> None:
        self.variables = variables
        self.constraints = constraints

    def is_satisfied(self, assignment: dict[CSPVariable, Any]) -> bool:
        return all(c.check(assignment) for c in self.constraints)


class FunctionalCSPSampler:
    """A sampler that generates candidate assignments."""

    def __init__(
        self, fn: Any, problem: CSP, variables: set[CSPVariable]
    ) -> None:
        self.fn = fn
        self.problem = problem
        self.variables = variables

    def sample(
        self, current: dict[CSPVariable, Any], rng: np.random.Generator
    ) -> dict[CSPVariable, Any]:
        return self.fn(current, rng)


class RandomWalkCSPSolver:
    """Solve a CSP by random walk."""

    def __init__(
        self,
        seed: int = 0,
        max_iters: int = 100,
        show_progress_bar: bool = False,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._max_iters = max_iters

    def solve(
        self,
        problem: CSP,
        initialization: dict[CSPVariable, Any],
        samplers: list[FunctionalCSPSampler],
    ) -> dict[CSPVariable, Any] | None:
        current = dict(initialization)
        if problem.is_satisfied(current):
            return current
        for _ in range(self._max_iters):
            for sampler in samplers:
                candidate = sampler.sample(current, self._rng)
                merged = {**current, **candidate}
                if problem.is_satisfied(merged):
                    return merged
                current = merged
        return None


class _CSPModule:
    """Bundle CSP classes into a module-like namespace for primitives['csp']."""

    CSPVariable = CSPVariable
    FunctionalCSPConstraint = FunctionalCSPConstraint
    CSP = CSP
    FunctionalCSPSampler = FunctionalCSPSampler
    RandomWalkCSPSolver = RandomWalkCSPSolver


_csp_module = _CSPModule()


# ── Counting wrapper ─────────────────────────────────────────────────────────


class _CountingEnv(gym.Wrapper):
    """Counts env.reset() calls == number of trajectories tried."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.reset_count = 0

    def reset(self, **kwargs: Any) -> Any:
        self.reset_count += 1
        return self.env.reset(**kwargs)


# ── Approach runner ──────────────────────────────────────────────────────────


def _run_approach_episode(
    env: gym.Env,
    monitor: FailureMonitor,
    approach: Any,
    seed: int,
    max_steps: int,
) -> bool:
    """Run one episode with a GeneratedApproach. Returns True if failure found."""
    wrapped = FailureMonitorWrapper(env, monitor)
    obs, info = wrapped.reset(seed=seed)
    approach.reset(obs, info)

    for _ in range(max_steps):
        action = approach.get_action(obs)
        next_obs, reward, terminated, truncated, info = wrapped.step(action)
        # Call update if approach supports it (blocks approach does)
        if hasattr(approach, "update"):
            approach.update(next_obs, reward, terminated or truncated, info)
        if reward == 1.0:
            return True
        if terminated or truncated:
            break
        obs = next_obs
    return False


def run(
    label: str,
    seeds: list[int],
    make_env_monitor: Any,
    make_approach: Any,
    max_steps: int,
    max_trajectories: int = 50,
) -> None:
    """Run an approach over seeds and print mean/std of trajectory counts."""
    counts: list[float] = []
    for seed in seeds:
        env, monitor = make_env_monitor()
        counting = _CountingEnv(env)
        counting.reset_count = 0
        approach = make_approach(env)

        found = False
        rng = np.random.default_rng(seed)
        for attempt in range(max_trajectories):
            episode_seed = int(rng.integers(0, 2**31))
            if _run_approach_episode(
                counting, monitor, approach, episode_seed, max_steps
            ):
                found = True
                break

        counts.append(
            float(counting.reset_count) if found else float("nan")
        )
        counting.close()

    arr = np.array(counts)
    valid = arr[~np.isnan(arr)]
    n_miss = int(np.sum(np.isnan(arr)))
    if len(valid) == 0:
        print(
            f"{label:<55}  "
            f"mean=   N/A  std=  N/A  min=N/A  max= N/A  "
            f"no_failure={n_miss}/{len(seeds)}"
        )
    else:
        print(
            f"{label:<55}  "
            f"mean={np.mean(valid):6.2f}  "
            f"std={np.std(valid):5.2f}  "
            f"min={int(np.min(valid)):3d}  "
            f"max={int(np.max(valid)):4d}  "
            f"no_failure={n_miss}/{len(seeds)}"
        )


# ── Primitives factory ───────────────────────────────────────────────────────


def _make_primitives(env: gym.Env) -> dict[str, Any]:
    """Build the primitives dict expected by GeneratedApproach classes."""

    def check_action_collision(obs: Any, action: Any) -> bool:
        """Stub: always returns False (no collision detected)."""
        return False

    return {
        "csp": _csp_module,
        "BiRRT": BiRRT,
        "check_action_collision": check_action_collision,
    }


# ── Approach factories ───────────────────────────────────────────────────────

SEEDS: list[int] = np.linspace(0, 1031, 14, dtype=int).tolist()


def main() -> None:
    """Run Claude-generated approaches and print trajectory count statistics."""
    print(f"Seeds: {SEEDS}\n")
    cols = f"{'mean':>8}  {'std':>7}  {'min':>5}  {'max':>6}  no_failure"
    header = f"{'Experiment':<55}  {cols}"
    print(header)
    print("-" * len(header))

    # ── Claude Blocks Approach ───────────────────────────────────────────────
    from gym_failure_discovery.failure_finders.claude_blocks_approach import (
        GeneratedApproach as BlocksApproach,
    )

    def _blocks_env_monitor():
        spec = BlocksSceneSpec(num_blocks=4, safe_height=0.15)
        raw = BlocksEnv(spec)
        return raw, BlocksFailureMonitor(raw)

    def _make_blocks_approach(env: gym.Env) -> BlocksApproach:
        primitives = _make_primitives(env)
        return BlocksApproach(
            env.action_space, env.observation_space, primitives
        )

    run(
        "Claude Approach / Blocks",
        SEEDS,
        _blocks_env_monitor,
        _make_blocks_approach,
        max_steps=20,
        max_trajectories=50,
    )

    # ── Claude ConveyorBelt Approach ─────────────────────────────────────────
    from gym_failure_discovery.failure_finders.claude_conveyorbelt_approach import (
        GeneratedApproach as ConveyorApproach,
    )

    def _conveyor_env_monitor():
        return (
            ConveyorBeltEnv(ConveyorBeltSceneSpec()),
            ConveyorBeltFailureMonitor(),
        )

    def _make_conveyor_approach(env: gym.Env) -> ConveyorApproach:
        primitives = _make_primitives(env)
        return ConveyorApproach(
            env.action_space, env.observation_space, primitives
        )

    run(
        "Claude Approach / ConveyorBelt",
        SEEDS,
        _conveyor_env_monitor,
        _make_conveyor_approach,
        max_steps=200,
        max_trajectories=50,
    )

    # ── Claude Hovercraft Approach ───────────────────────────────────────────
    from gym_failure_discovery.failure_finders.claude_hovercraft_approach import (
        GeneratedApproach as HovercraftApproach,
    )

    def _hovercraft_env_monitor():
        return (
            HoverCraftEnv(HoverCraftSceneSpec()),
            HoverCraftFailureMonitor(HoverCraftSceneSpec()),
        )

    def _make_hovercraft_approach(env: gym.Env) -> HovercraftApproach:
        primitives = _make_primitives(env)
        return HovercraftApproach(
            env.action_space, env.observation_space, primitives
        )

    run(
        "Claude Approach / Hovercraft",
        SEEDS,
        _hovercraft_env_monitor,
        _make_hovercraft_approach,
        max_steps=500,
        max_trajectories=50,
    )


if __name__ == "__main__":
    main()
