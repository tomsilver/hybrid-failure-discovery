"""Tests for hovercraft_controller.py."""

from hybrid_failure_discovery.controllers.hovercraft_controller import (
    HoverCraftParameterizedController,
)
from hybrid_failure_discovery.envs.hovercraft_env import (
    HoverCraftEnv,
)


def test_hovercraft_controller():
    """Tests for hovercraft_controller.py."""

    env = HoverCraftEnv()
    parameterized_controller = HoverCraftParameterizedController(env.scene_spec)

    # Uncomment to create video.
    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/test-hovercraft-controller")

    state, _ = env.reset(seed=123)
    parameterized_controller.reset(state)

    for _ in range(100):
        parameterized_action = True
        action = parameterized_controller.step(state, parameterized_action)
        state, _, _, _, _ = env.step(action)

    env.close()
