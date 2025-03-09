"""Tests for blocks_controller.py."""

from hybrid_failure_discovery.controllers.blocks_controller import (
    BlocksCommand,
    BlocksController,
)
from hybrid_failure_discovery.envs.blocks_env import (
    BlocksEnv,
)


def test_blocks_controller():
    """Tests for blocks_controller.py."""

    env = BlocksEnv(seed=123, use_gui=False)
    controller = BlocksController(123, env.scene_spec, safe_height=0.15)

    # Uncomment to create video.
    from gymnasium.wrappers import RecordVideo
    env = RecordVideo(env, "videos/test-blocks-controller")

    state, _ = env.reset(seed=123)
    controller.reset(state)
    command = BlocksCommand([["block1", "block2"]])

    for _ in range(200):
        action = controller.step(state, command)
        state, _, _, _, _ = env.step(action)

    env.close()
