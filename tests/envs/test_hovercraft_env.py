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

    # Uncomment to make videos
    # from pathlib import Path
    # from gymnasium.wrappers import RecordVideo
    # video_dir = Path("videos/test-hovercraft-env")
    # video_dir.mkdir(parents=True, exist_ok=True)

    for trial in range(5):  # pylint: disable=unused-variable
        # Uncomment to make videos
        # env = RecordVideo(
        #     env,
        #     str(video_dir / f"trial-{trial}"),
        #     episode_trigger=lambda episode_id: episode_id == 0,
        # )

        state, _ = env.reset(seed=123)
        assert isinstance(state, HoverCraftState)
        rng = np.random.default_rng(123)

        for _ in range(25):
            ux, uy = rng.normal(size=2)
            action = HoverCraftAction(float(ux), float(uy))
            state, _, _, _, _ = env.step(action)
            assert isinstance(state, HoverCraftState)

        env.close()
