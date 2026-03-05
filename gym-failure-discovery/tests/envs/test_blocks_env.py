"""Tests for the blocks environment."""

import numpy as np
import pytest

from gym_failure_discovery.envs.blocks_env import (
    PICK,
    STACK,
    UNSTACK,
    BlocksEnv,
    BlocksSceneSpec,
)
from gym_failure_discovery.failure_monitors.blocks_failure_monitor import (
    BlocksFailureMonitor,
)


def _pick(block: int) -> dict:
    return {"type": PICK, "block": block}


def _unstack(block: int) -> dict:
    return {"type": UNSTACK, "block": block}


def _stack(block: int) -> dict:
    return {"type": STACK, "block": block}


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
    obs, _, _, _, _ = env.step(_pick(0))
    assert obs["held_block"] == 0
    env.close()


def test_pick_and_stack():
    """Pick a block, then stack it on another block."""
    spec = BlocksSceneSpec(num_blocks=3)
    env = BlocksEnv(spec)
    obs, _ = env.reset(seed=0)
    target_pos_before = obs["block_positions"][1].copy()

    obs, _, _, _, _ = env.step(_pick(0))
    assert obs["held_block"] == 0

    obs, _, _, _, _ = env.step(_stack(1))
    assert obs["held_block"] == 3  # nothing held
    assert obs["block_positions"][0, 2] > target_pos_before[2] + 0.01
    env.close()


def test_unstack_block():
    """Unstack should work the same as pick for a stacked block."""
    spec = BlocksSceneSpec(num_blocks=3)
    env = BlocksEnv(spec)
    env.reset(seed=0)

    env.step(_pick(0))
    env.step(_stack(1))

    obs, _, _, _, _ = env.step(_unstack(0))
    assert obs["held_block"] == 0
    env.close()


def test_stack_without_holding_is_noop():
    """Stack when nothing is held should be a no-op."""
    spec = BlocksSceneSpec(num_blocks=3)
    env = BlocksEnv(spec)
    obs_before, _ = env.reset(seed=0)
    obs_after, _, _, _, _ = env.step(_stack(1))
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
        env.step(_pick(0))
        state = env.get_state()
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


def test_frame_buffer():
    """Intermediate frames are buffered when render_mode is set."""
    spec = BlocksSceneSpec(num_blocks=3)
    env = BlocksEnv(spec, render_mode="rgb_array")
    env.reset(seed=0)
    env.step(_pick(0))
    frames = env.pop_frame_buffer()
    assert len(frames) > 1
    assert frames[0].shape[2] == 3
    env.close()


@pytest.mark.make_videos
def test_blocks_pick_and_stack_video(maybe_record):  # type: ignore
    """Pick and stack several blocks, recording a video."""
    spec = BlocksSceneSpec(num_blocks=4)
    env = maybe_record(BlocksEnv(spec))
    env.reset(seed=0)
    env.step(_pick(0))
    env.step(_stack(1))
    env.step(_pick(2))
    env.step(_stack(0))
    env.close()


@pytest.mark.make_videos
def test_blocks_tall_tower_failure(maybe_record):  # type: ignore
    """Build a tall tower with low safe_height to elicit a collision."""
    spec = BlocksSceneSpec(num_blocks=6, safe_height=0.15)
    env = BlocksEnv(spec)
    wrapped = maybe_record(env)
    monitor = BlocksFailureMonitor(env)
    obs, _ = wrapped.reset(seed=0)
    monitor.reset(obs)

    failure = False
    for i in range(1, spec.num_blocks):
        for action in [_pick(i), _stack(i - 1)]:
            prev_obs = obs
            obs, _, _, _, _ = wrapped.step(action)
            if monitor.step(prev_obs, action, obs):
                failure = True
                break
        if failure:
            break

    assert failure, "Expected a collision building a tall tower with safe_height=0.15"
    wrapped.close()
