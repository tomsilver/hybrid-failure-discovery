"""Tests for the hovercraft environment."""

import numpy as np

from gym_failure_discovery.hovercraft_env import HoverCraftEnv


def test_reset_and_step():
    """Basic reset/step cycle with random actions."""
    env = HoverCraftEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (5,)
    assert not info

    for _ in range(50):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        assert obs.shape == (5,)
        assert not terminated
        assert not truncated

    env.close()


def test_deterministic_reset():
    """Same seed produces the same initial observation."""
    env = HoverCraftEnv()
    obs1, _ = env.reset(seed=7)
    obs2, _ = env.reset(seed=7)
    np.testing.assert_array_equal(obs1, obs2)
    env.close()


def test_switching_changes_trajectory():
    """Switching vs not switching should produce different trajectories."""
    env = HoverCraftEnv()

    env.reset(seed=0)
    for _ in range(20):
        env.step(0)
    obs_no_switch, *_ = env.step(0)

    env.reset(seed=0)
    for _ in range(20):
        env.step(0)
    obs_switch, *_ = env.step(1)

    assert not np.allclose(obs_no_switch, obs_switch)
    env.close()


def test_render():
    """Render returns an image array."""
    env = HoverCraftEnv()
    env.reset(seed=0)
    img = env.render()
    assert img is not None
    assert len(img.shape) == 3  # H x W x C
    env.close()


def test_observation_space_contains_obs():
    """Observations should be within the declared observation space."""
    env = HoverCraftEnv()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)

    for _ in range(10):
        obs, *_ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)

    env.close()
