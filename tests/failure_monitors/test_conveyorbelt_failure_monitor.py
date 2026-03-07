"""Tests for the conveyor belt failure monitor."""

from gym_failure_discovery.envs.conveyorbelt_env import (
    FAST,
    ConveyorBeltEnv,
)
from gym_failure_discovery.failure_monitors.conveyorbelt_failure_monitor import (
    ConveyorBeltFailureMonitor,
)


def test_no_failure_without_secret_sequence():
    """Normal operation should not trigger a failure."""
    env = ConveyorBeltEnv()
    monitor = ConveyorBeltFailureMonitor()
    obs, _ = env.reset(seed=0)
    monitor.reset(obs)
    for _ in range(100):
        prev = obs
        obs, _, _, _, _ = env.step(FAST)
        assert not monitor.step(prev, FAST, obs)
