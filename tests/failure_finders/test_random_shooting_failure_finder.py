"""Tests for the random shooting failure finder."""

import pytest

from gym_failure_discovery.envs.blocks_env import BlocksEnv, BlocksSceneSpec
from gym_failure_discovery.envs.conveyorbelt_env import (
    ConveyorBeltEnv,
    ConveyorBeltSceneSpec,
)
from gym_failure_discovery.envs.hovercraft_env import HoverCraftEnv, HoverCraftSceneSpec
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


@pytest.mark.make_videos
def test_random_shooting_finds_hovercraft_failure(maybe_record):  # type: ignore
    """Random shooting should find a collision in the hovercraft env."""
    spec = HoverCraftSceneSpec()
    env = maybe_record(HoverCraftEnv(spec))
    monitor = HoverCraftFailureMonitor(spec)
    finder = RandomShootingFailureFinder(
        seed=0, max_num_trajectories=50, max_trajectory_length=200
    )
    result = finder.find_failure(env, monitor)
    assert result is not None
    assert len(result) > 0
    env.close()

# added -- for conveyorbelt
@pytest.mark.make_videos
def test_random_shooting_finds_conveyorbelt_failure(maybe_record):  # type: ignore
    """Random shooting should find the secret sequence in the conveyor belt env."""
    spec = ConveyorBeltSceneSpec()
    raw_env = ConveyorBeltEnv(spec, render_mode="rgb_array")
    monitor = ConveyorBeltFailureMonitor()

    # Search without recording to avoid one video per trajectory attempt.
    finder = RandomShootingFailureFinder(
        seed=0, max_num_trajectories=1000, max_trajectory_length=200
    )
    result = finder.find_failure(raw_env, monitor)
    assert result is not None
    assert len(result) > 0

    # Replay only the failure trajectory with the video wrapper.
    # ConveyorBeltEnv always resets to an empty belt, so the same actions
    # reproduce the identical trajectory regardless of seed.
    env = maybe_record(raw_env)
    env.reset(seed=0)
    for _, action in result:
        env.step(action)
    env.close()


# added -- for blocks env
@pytest.mark.make_videos
def test_random_shooting_finds_blocks_failure(maybe_record):  # type: ignore
    """Random shooting should find a collision in the blocks env."""
    spec = BlocksSceneSpec(num_blocks=6, safe_height=0.15)
    raw_env = BlocksEnv(spec)
    monitor = BlocksFailureMonitor(raw_env)
    env = maybe_record(raw_env)
    finder = RandomShootingFailureFinder(
        seed=0, max_num_trajectories=50, max_trajectory_length=10
    )
    result = finder.find_failure(env, monitor)
    assert result is not None
    assert len(result) > 0
    env.close()
