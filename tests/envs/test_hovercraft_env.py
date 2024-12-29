"""Tests for hovercraft_env.py."""

import numpy as np

from hybrid_failure_discovery.envs.hovercraft_env import (
    HoverCraftAction,
    HoverCraftEnv,
    HoverCraftState,
)


def test_hovercraft_env():
    """Tests for hovercraft_env.py."""

    env = HoverCraftEnv()

    # Uncomment to create video.
    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/test-hovercraft-env")

    state, _ = env.reset(seed=123)
    assert isinstance(state, HoverCraftState)
    rng = np.random.default_rng(123)

    for _ in range(25):
        ux, uy = rng.normal(size=2)
        action = HoverCraftAction(float(ux), float(uy))
        state, _, _, _, _ = env.step(action)
        assert isinstance(state, HoverCraftState)

    env.close()
