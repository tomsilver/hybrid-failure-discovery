"""Tests for conveyorbelt_failure_monitor.py."""

from hybrid_failure_discovery.envs.conveyorbelt_env import (
    ConveyorBeltEnv,
    ConveyorBeltSceneSpec,
)
from hybrid_failure_discovery.failure_monitors.conveyorbelt_failure_monitor import (
    ConveyorBeltFailureMonitor,
)


def test_conveyorbelt_failure_monitor():
    """Tests for conveyorbelt_failure_monitor.py."""
    scene_spec = ConveyorBeltSceneSpec()
    monitor = ConveyorBeltFailureMonitor(scene_spec)

    # An initial state should be far from failure and so have a high score.
    env = ConveyorBeltEnv(scene_spec, seed=123)
    initial_state, _ = env.reset(seed=123)
    score = monitor.get_robustness_score(initial_state)
    print("Initial robustness score: ", score)
