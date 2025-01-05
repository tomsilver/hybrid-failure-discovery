"""Tests for hovercraft_controller.py."""

from hybrid_failure_discovery.controllers.hovercraft_controller import (
    HoverCraftController,
)
from hybrid_failure_discovery.envs.hovercraft_env import (
    HoverCraftEnv,
)


def test_hovercraft_controller():
    """Tests for hovercraft_controller.py."""

    env = HoverCraftEnv()
    controller = HoverCraftController(env.scene_spec)

    # Uncomment to create video.
    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/test-hovercraft-controller")

    state, _ = env.reset(seed=123)
    controller.reset(state)

    for _ in range(250):
        action = controller.step(state)
        state, _, _, _, _ = env.step(action)

    env.close()
