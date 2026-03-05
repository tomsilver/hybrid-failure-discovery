"""Tests for the random shooting failure finder."""

from gym_failure_discovery.envs.hovercraft_env import HoverCraftEnv, HoverCraftSceneSpec
from gym_failure_discovery.failure_finders.random_shooting_failure_finder import (
    RandomShootingFailureFinder,
)
from gym_failure_discovery.failure_monitors.hovercraft_failure_monitor import (
    HoverCraftFailureMonitor,
)


def test_random_shooting_finds_hovercraft_failure():
    """Random shooting should find a collision in the hovercraft env."""
    spec = HoverCraftSceneSpec()
    env = HoverCraftEnv(spec)
    monitor = HoverCraftFailureMonitor(spec)
    finder = RandomShootingFailureFinder(
        seed=0, max_num_trajectories=50, max_trajectory_length=200
    )
    result = finder.find_failure(env, monitor)
    assert result is not None
    assert len(result) > 0
    env.close()
