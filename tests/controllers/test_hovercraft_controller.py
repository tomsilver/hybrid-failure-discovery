"""Tests for hovercraft_controller.py."""

from hybrid_failure_discovery.controllers.hovercraft_controller import (
    HoverCraftController,
)
from hybrid_failure_discovery.envs.hovercraft_env import (
    HoverCraftCommand,
    HoverCraftEnv,
)


def test_hovercraft_controller():
    """Tests for hovercraft_controller.py."""

    env = HoverCraftEnv()
    controller = HoverCraftController(123, env.scene_spec)

    # Uncomment to create video.
    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/test-hovercraft-controller")

    state, _ = env.reset(seed=123)
    controller.reset(state)

    for t in range(250):
        switch = t == 100
        command = HoverCraftCommand(switch)
        action = controller.step(state, command)
        state, _, _, _, _ = env.step(action)

    env.close()
