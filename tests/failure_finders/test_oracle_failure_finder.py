"""Tests for the oracle failure finder."""

from typing import Any

import numpy as np
import pytest

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
from gym_failure_discovery.failure_finders.oracle_failure_finder import (
    OracleFailureFinder,
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


class _SwitchNearGoalPolicy(Policy):
    """Switch goal pair once when near the right goal."""

    def __init__(self, spec: HoverCraftSceneSpec) -> None:
        self._goal_x = spec.goal_pairs[0][1][0]
        self._switched = False

    def reset(self) -> None:
        self._switched = False

    def act(self, obs: np.ndarray) -> int:
        current_x = obs[0]
        if abs(current_x - self._goal_x) < 0.1 and not self._switched:
            self._switched = True
            return 1
        return 0


class _NoOpPolicy(Policy):
    """Always returns action 0."""

    def act(self, obs: np.ndarray) -> int:
        return 0


@pytest.mark.make_videos
def test_oracle_finds_failure_with_switch_policy(maybe_record):  # type: ignore
    """Switching near the right goal should cause a collision."""
    spec = HoverCraftSceneSpec()
    monitor = HoverCraftFailureMonitor(spec)
    policy = _SwitchNearGoalPolicy(spec)
    env = maybe_record(HoverCraftEnv(spec))
    oracle = OracleFailureFinder(policy=policy, seed=0, max_trajectory_length=500)
    result = oracle.find_failure(env, monitor)
    assert result is not None
    assert len(result) > 0
    env.close()


class _TowerBuildingPolicy(Policy):
    """Builds a tower by sequentially picking block i and stacking on block i-1."""

    def __init__(self, num_blocks: int) -> None:
        self._num_blocks = num_blocks
        self._action_queue: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._action_queue = []
        for i in range(1, self._num_blocks):
            self._action_queue.append({"type": PICK, "block": i})
            self._action_queue.append({"type": STACK, "block": i - 1})

    def act(self, obs: Any) -> dict[str, Any]:
        if self._action_queue:
            return self._action_queue.pop(0)
        return {"type": STACK, "block": 0}


@pytest.mark.make_videos
def test_oracle_finds_blocks_failure(maybe_record):  # type: ignore
    """Tower-building policy should cause a collision with low safe height."""
    spec = BlocksSceneSpec(num_blocks=4, safe_height=0.15)
    raw_env = BlocksEnv(spec)
    monitor = BlocksFailureMonitor(raw_env)
    policy = _TowerBuildingPolicy(spec.num_blocks)
    env = maybe_record(raw_env)
    oracle = OracleFailureFinder(policy=policy, seed=0, max_trajectory_length=20)
    result = oracle.find_failure(env, monitor)
    assert result is not None
    assert len(result) > 0
    env.close()


class _SecretSequencePolicy(Policy):
    """Issues the known failure-inducing sequence to trigger the explosion."""

    def __init__(self) -> None:
        self._step: int = 0

    def reset(self) -> None:
        self._step = 0

    def act(self, obs: Any) -> int:
        sequence = [FAST, SLOW, FAST, MID, FAST, FAST, FAST, SLOW]
        if self._step < len(sequence):
            action = sequence[self._step]
            self._step += 1
            return action
        return OFF


@pytest.mark.make_videos
def test_oracle_finds_conveyorbelt_failure(maybe_record):  # type: ignore
    """Issuing the known failure sequence should trigger the explosion."""
    spec = ConveyorBeltSceneSpec()
    env = maybe_record(ConveyorBeltEnv(spec, render_mode="rgb_array"))
    monitor = ConveyorBeltFailureMonitor()
    policy = _SecretSequencePolicy()
    oracle = OracleFailureFinder(policy=policy, seed=0, max_trajectory_length=13)
    result = oracle.find_failure(env, monitor)
    assert result is not None
    assert len(result) > 0
    env.close()


def test_oracle_returns_none_for_safe_policy():
    """A no-op policy over a short horizon should not produce a failure."""
    spec = HoverCraftSceneSpec()
    monitor = HoverCraftFailureMonitor(spec)
    oracle = OracleFailureFinder(policy=_NoOpPolicy(), seed=42, max_trajectory_length=5)
    result = oracle.find_failure(HoverCraftEnv(spec), monitor)
    assert result is None
