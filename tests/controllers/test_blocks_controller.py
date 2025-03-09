"""Tests for blocks_controller.py."""

from relational_structs import GroundAtom

from hybrid_failure_discovery.controllers.blocks_controller import (
    BlocksCommand,
    BlocksController,
    BlocksPerceiver,
    GripperEmpty,
    On,
    object_type,
    robot_type,
)
from hybrid_failure_discovery.envs.blocks_env import (
    BlocksEnv,
)


def test_blocks_controller():
    """Tests for blocks_controller.py."""

    env = BlocksEnv(seed=123, use_gui=False)
    controller = BlocksController(123, env.scene_spec, safe_height=0.2)
    perceiver = BlocksPerceiver(BlocksEnv(env.scene_spec))

    # Uncomment to create video.
    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/test-blocks-controller")

    state, _ = env.reset(seed=123)
    controller.reset(state)
    command = BlocksCommand([["block1", "block2"]])
    perceiver.reset(state, {"goal": command})
    robot = robot_type("robot")
    block1, block2 = object_type("block1"), object_type("block2")
    goal = {GroundAtom(On, [block2, block1]), GroundAtom(GripperEmpty, [robot])}

    for _ in range(200):
        action = controller.step(state, command)
        state, _, _, _, _ = env.step(action)
        atoms = perceiver.step(state)
        if goal.issubset(atoms):
            break
    else:
        assert False, "Goal not reached"

    env.close()
