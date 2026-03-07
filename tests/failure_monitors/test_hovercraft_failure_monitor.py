"""Tests for failure monitors and the wrapper."""

import numpy as np

from gym_failure_discovery.envs.hovercraft_env import HoverCraftEnv, HoverCraftSceneSpec
from gym_failure_discovery.failure_monitor_wrapper import FailureMonitorWrapper
from gym_failure_discovery.failure_monitors.hovercraft_failure_monitor import (
    HoverCraftFailureMonitor,
)


def test_hovercraft_failure_monitor_detects_collision():
    """A state inside an obstacle should be detected as a failure."""
    spec = HoverCraftSceneSpec()
    monitor = HoverCraftFailureMonitor(spec)
    # Obstacle at bottom-left: Rectangle(-0.5, -0.5, 0.3, 0.3)
    inside_obstacle = np.array([-0.35, 0.0, -0.35, 0.0, 0.0])
    safe_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    monitor.reset(safe_state)
    assert monitor.step(safe_state, 0, inside_obstacle)
    assert not monitor.step(safe_state, 0, safe_state)


def test_wrapper_reward_on_failure():
    """Wrapper should give reward=1 and terminate on failure."""
    spec = HoverCraftSceneSpec()
    monitor = HoverCraftFailureMonitor(spec)
    env = FailureMonitorWrapper(HoverCraftEnv(spec), monitor)

    env.reset(seed=0)
    for _ in range(10):
        _, reward, terminated, _, _ = env.step(0)
        if terminated:
            assert reward == 1.0
            break
        assert reward == 0.0

    env.close()


def test_wrapper_no_failure_gives_zero_reward():
    """Without failure, all rewards should be 0."""
    spec = HoverCraftSceneSpec()
    monitor = HoverCraftFailureMonitor(spec)
    env = FailureMonitorWrapper(HoverCraftEnv(spec), monitor)

    env.reset(seed=42)
    rewards = []
    for _ in range(20):
        _, reward, terminated, _, _ = env.step(0)
        rewards.append(reward)
        if terminated:
            break

    assert len(rewards) > 0
    for r in rewards[:-1]:
        assert r == 0.0

    env.close()
