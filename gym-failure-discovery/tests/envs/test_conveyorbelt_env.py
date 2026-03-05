"""Tests for the conveyor belt environment."""

import pytest

from gym_failure_discovery.envs.conveyorbelt_env import (
    FAST,
    MID,
    OFF,
    SLOW,
    ConveyorBeltEnv,
    ConveyorBeltSceneSpec,
)
from gym_failure_discovery.failure_monitors.conveyorbelt_failure_monitor import (
    ConveyorBeltFailureMonitor,
)


def test_reset_empty_belt():
    """After reset, belt should be empty and not exploded."""
    env = ConveyorBeltEnv()
    obs, _ = env.reset(seed=0)
    assert len(obs["positions"]) == 0
    assert not obs["exploded"]


def test_off_mode_never_drops():
    """Off mode should never drop packages."""
    env = ConveyorBeltEnv()
    env.reset(seed=0)
    for _ in range(500):
        obs, _, _, _, _ = env.step(OFF)
    assert len(obs["positions"]) == 0


def test_fast_mode_drops_packages():
    """Fast mode should drop packages onto the belt."""
    env = ConveyorBeltEnv()
    env.reset(seed=0)
    for _ in range(100):
        obs, _, _, _, _ = env.step(FAST)
    assert len(obs["positions"]) > 0


def test_secret_sequence_triggers_explosion():
    """Issuing the secret sequence should explode the belt."""
    spec = ConveyorBeltSceneSpec(secret_sequence=[SLOW, FAST, MID])
    env = ConveyorBeltEnv(spec)
    env.reset(seed=0)
    for mode in [SLOW, FAST, MID]:
        obs, _, terminated, _, _ = env.step(mode)
    assert obs["exploded"]
    assert terminated


def test_wrong_sequence_does_not_explode():
    """A different sequence should not trigger explosion."""
    spec = ConveyorBeltSceneSpec(secret_sequence=[SLOW, FAST, MID])
    env = ConveyorBeltEnv(spec)
    env.reset(seed=0)
    for mode in [FAST, SLOW, MID]:
        obs, _, terminated, _, _ = env.step(mode)
    assert not obs["exploded"]
    assert not terminated


def test_failure_monitor_detects_explosion():
    """The failure monitor should flag the explosion."""
    spec = ConveyorBeltSceneSpec(secret_sequence=[SLOW, FAST])
    env = ConveyorBeltEnv(spec)
    monitor = ConveyorBeltFailureMonitor()
    obs, _ = env.reset(seed=0)
    monitor.reset(obs)

    prev = obs
    obs, _, _, _, _ = env.step(SLOW)
    assert not monitor.step(prev, SLOW, obs)

    prev = obs
    obs, _, _, _, _ = env.step(FAST)
    assert monitor.step(prev, FAST, obs)


def test_packages_move_along_belt():
    """Packages on the belt should advance with belt velocity."""
    spec = ConveyorBeltSceneSpec(drop_start_height=0.0)
    env = ConveyorBeltEnv(spec)
    env.reset(seed=0)
    # Drop a package (height=0 so it's on the belt immediately).
    env.step(FAST)
    obs_before, _, _, _, _ = env.step(OFF)
    pos_before = obs_before["positions"][0]
    for _ in range(100):
        obs_after, _, _, _, _ = env.step(OFF)
    assert obs_after["positions"][0] > pos_before


@pytest.mark.make_videos
def test_conveyorbelt_video(maybe_record):  # type: ignore
    """Run the belt with fast drops, recording a video."""
    env = maybe_record(ConveyorBeltEnv())
    env.reset(seed=0)
    for _ in range(200):
        env.step(FAST)
    env.close()
