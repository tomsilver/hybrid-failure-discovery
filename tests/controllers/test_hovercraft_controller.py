"""Tests for hovercraft_controller.py."""

from hybrid_failure_discovery.envs.hovercraft_env import (
    HoverCraftEnv,
)
from hybrid_failure_discovery.controllers.hovercraft_controller import HoverCraftController


def test_hovercraft_controller():
    """Tests for hovercraft_controller.py."""

    env = HoverCraftEnv()
    controller = HoverCraftController(env.scene_spec, seed=123)

    # Uncomment to create video.
    # TODO
    from gymnasium.wrappers import RecordVideo
    env = RecordVideo(env, "videos/test-hovercraft-controller")

    state, info = env.reset(seed=123)
    controller.reset(state, info)

    for _ in range(25):
        action = controller.step()
        state, reward, done, _, info = env.step(action)
        controller.update(state, reward, done, info)

    env.close()
