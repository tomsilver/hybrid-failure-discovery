"""Tests for the blocks environment."""

import numpy as np

from gym_failure_discovery.envs.blocks_env import (
    PICK,
    STACK,
    UNSTACK,
    BlocksEnv,
    BlocksSceneSpec,
)


def test_reset_places_blocks_on_table():
    """After reset, all blocks should be on the table and nothing held."""
    spec = BlocksSceneSpec(num_blocks=3)
    env = BlocksEnv(spec)
    obs, _ = env.reset(seed=0)
    assert obs["block_positions"].shape == (3, 3)
    assert obs["held_block"] == 3  # nothing held
    table_z = spec.table_pose.position[2] + spec.table_half_extents[2]
    for i in range(3):
        assert obs["block_positions"][i, 2] > table_z - 0.01
    env.close()


def test_pick_block():
    """Pick should grasp a block from the table."""
    spec = BlocksSceneSpec(num_blocks=3)
    env = BlocksEnv(spec)
    env.reset(seed=0)
    obs, _, _, _, _ = env.step({"type": PICK, "block": 0})
    assert obs["held_block"] == 0
    env.close()


def test_pick_and_stack():
    """Pick a block, then stack it on another block."""
    spec = BlocksSceneSpec(num_blocks=3)
    env = BlocksEnv(spec)
    obs, _ = env.reset(seed=0)
    target_pos_before = obs["block_positions"][1].copy()

    # Pick block 0.
    obs, _, _, _, _ = env.step({"type": PICK, "block": 0})
    assert obs["held_block"] == 0

    # Stack onto block 1.
    obs, _, _, _, _ = env.step({"type": STACK, "block": 1})
    assert obs["held_block"] == 3  # nothing held
    # Block 0 should now be above block 1.
    assert obs["block_positions"][0, 2] > target_pos_before[2] + 0.01
    env.close()


def test_unstack_block():
    """Unstack should work the same as pick for a stacked block."""
    spec = BlocksSceneSpec(num_blocks=3)
    env = BlocksEnv(spec)
    env.reset(seed=0)

    # First stack: pick 0, stack on 1.
    env.step({"type": PICK, "block": 0})
    env.step({"type": STACK, "block": 1})

    # Unstack block 0 from block 1.
    obs, _, _, _, _ = env.step({"type": UNSTACK, "block": 0})
    assert obs["held_block"] == 0
    env.close()


def test_stack_without_holding_is_noop():
    """Stack when nothing is held should be a no-op."""
    spec = BlocksSceneSpec(num_blocks=3)
    env = BlocksEnv(spec)
    obs_before, _ = env.reset(seed=0)
    obs_after, _, _, _, _ = env.step({"type": STACK, "block": 1})
    assert obs_after["held_block"] == 3
    np.testing.assert_allclose(
        obs_before["block_positions"],
        obs_after["block_positions"],
        atol=0.01,
    )
    env.close()


def test_safe_height_affects_trajectory():
    """Lower safe_height means the robot passes closer to other blocks."""
    for safe_h in [0.25, 0.05]:
        spec = BlocksSceneSpec(num_blocks=3, safe_height=safe_h)
        env = BlocksEnv(spec)
        env.reset(seed=0)
        env.step({"type": PICK, "block": 0})
        state = env.get_state()
        # The held block should have been lifted to around safe_height.
        held_name = env.block_names[state.held_block_idx]
        held_z = state.get_block_state(held_name).pose.position[2]
        assert abs(held_z - safe_h) < 0.1, f"safe_height={safe_h}, held_z={held_z}"
        env.close()


def test_render():
    """Render should return an RGB image."""
    spec = BlocksSceneSpec(num_blocks=2)
    env = BlocksEnv(spec)
    env.reset(seed=0)
    img = env.render()
    assert img is not None
    assert len(img.shape) == 3
    assert img.shape[2] == 3
    env.close()
