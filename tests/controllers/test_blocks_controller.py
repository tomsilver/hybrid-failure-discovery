"""Tests for blocks_controller.py."""

from hybrid_failure_discovery.controllers.blocks_controller import (
    BlocksController,
)
from hybrid_failure_discovery.envs.blocks_env import (
    BlocksEnv,
)


def test_blocks_controller():
    """Tests for blocks_controller.py."""

    env = BlocksEnv(seed=123, use_gui=True)
    controller = BlocksController(123, env.scene_spec)

    # Uncomment to create video.
    from gymnasium.wrappers import RecordVideo

    env = RecordVideo(env, "videos/test-blocks-controller")

    state, _ = env.reset(seed=123)
    controller.reset(state)

    for _ in range(100):
        action = controller.step(state)
        state, _, _, _, _ = env.step(action)

    env.close()
