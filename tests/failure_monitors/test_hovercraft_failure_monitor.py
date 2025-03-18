"""Tests for hovercraft_failure_monitor.py."""

from hybrid_failure_discovery.envs.hovercraft_env import (
    HoverCraftEnv,
    HoverCraftSceneSpec,
)
from hybrid_failure_discovery.failure_monitors.hovercraft_failure_monitor import (
    HoverCraftFailureMonitor,
)


def test_hovercraft_failure_monitor():
    """Tests for hovercraft_failure_monitor.py."""
    scene_spec = HoverCraftSceneSpec()
    monitor = HoverCraftFailureMonitor(scene_spec)

    # An initial state should be far from failure and so have a high score.
    env = HoverCraftEnv(scene_spec, seed=123)
    initial_state, _ = env.reset(seed=123)
    score = monitor.get_robustness_score(initial_state)
