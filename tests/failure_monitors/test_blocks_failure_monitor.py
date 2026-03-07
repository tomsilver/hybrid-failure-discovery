"""Tests for the blocks failure monitor."""

from gym_failure_discovery.envs.blocks_env import (
    PICK,
    STACK,
    BlocksEnv,
    BlocksSceneSpec,
)
from gym_failure_discovery.failure_monitors.blocks_failure_monitor import (
    BlocksFailureMonitor,
)


def test_no_failure_on_normal_stack():
    """Normal pick-and-stack should not trigger a failure."""
    spec = BlocksSceneSpec(num_blocks=3, safe_height=0.25)
    env = BlocksEnv(spec)
    monitor = BlocksFailureMonitor(env)
    obs, _ = env.reset(seed=0)
    monitor.reset(obs)

    obs2, _, _, _, _ = env.step({"type": PICK, "block": 0})
    assert not monitor.step(obs, {"type": PICK, "block": 0}, obs2)

    obs3, _, _, _, _ = env.step({"type": STACK, "block": 2})
    assert not monitor.step(obs2, {"type": STACK, "block": 2}, obs3)
    env.close()
