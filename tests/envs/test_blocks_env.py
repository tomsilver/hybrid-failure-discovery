"""Tests for blocks_env.py."""

import numpy as np

from hybrid_failure_discovery.envs.blocks_env import (
    BlocksAction,
    BlocksEnv,
    BlocksEnvState,
)


def test_blocks_env():
    """Tests for blocks_env.py."""

    env = BlocksEnv(seed=123, use_gui=False)
    env.robot.action_space.seed(123)

    # Uncomment to create video.
    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/test-blocks-env")

    state, _ = env.reset(seed=123)
    assert isinstance(state, BlocksEnvState)
    rng = np.random.default_rng(123)

    for _ in range(25):
        joint_action = env.robot.action_space.sample()[:7]
        gripper_action = rng.choice([-1, 0, 1])
        action = BlocksAction(joint_action, gripper_action)
        state, _, _, _, _ = env.step(action)
        assert isinstance(state, BlocksEnvState)

    env.close()
