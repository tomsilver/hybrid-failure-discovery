"""Tests for hovercraft_env.py."""

from hybrid_failure_discovery.envs.hovercraft_env import (
    HoverCraftEnv,
    HoverCraftAction,
    HoverCraftState,
)
import numpy as np


def test_hovercraft_env():
    """Tests for hovercraft_env.py."""

    env = HoverCraftEnv()

    # Uncomment to create video.
    # TODO comment out
    from gymnasium.wrappers import RecordVideo
    env = RecordVideo(env, "videos/test-hovercraft-env")

    state, _ = env.reset(seed=123)
    assert isinstance(state, HoverCraftState)
    rng = np.random.default_rng(123)

    for _ in range(100):
        ux, uy = rng.normal(size=2)
        action = HoverCraftAction(float(ux), float(uy))
        state, _, _, _, _ = env.step(action)
        assert isinstance(state, HoverCraftState)

    env.close()
