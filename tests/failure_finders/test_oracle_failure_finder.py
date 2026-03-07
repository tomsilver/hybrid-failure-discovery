"""Tests for the oracle failure finder."""

import numpy as np
import pytest

from gym_failure_discovery.envs.hovercraft_env import HoverCraftEnv, HoverCraftSceneSpec
from gym_failure_discovery.failure_finders.oracle_failure_finder import (
    OracleFailureFinder,
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


def test_oracle_returns_none_for_safe_policy():
    """A no-op policy over a short horizon should not produce a failure."""
    spec = HoverCraftSceneSpec()
    monitor = HoverCraftFailureMonitor(spec)
    oracle = OracleFailureFinder(policy=_NoOpPolicy(), seed=42, max_trajectory_length=5)
    result = oracle.find_failure(HoverCraftEnv(spec), monitor)
    assert result is None
