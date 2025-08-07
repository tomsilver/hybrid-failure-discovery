"""Unit tests for the ConveyorBelt environment.

This module tests the functionality of the ConveyorBeltEnv class, including:
- Valid state transitions based on discrete actions (no-op, forward, reverse)
- Proper video recording during environment interaction
- Correct velocity setting and *non-wrapped* position updates for boxes
- Automatic box dropping when space is available
- Box spacing correctness
"""

from pathlib import Path

import numpy as np
from gymnasium.wrappers import RecordVideo

from hybrid_failure_discovery.envs.conveyorbelt_env import (
    ConveyorBeltAction,
    ConveyorBeltEnv,
    ConveyorBeltState,
)


def test_conveyorbelt_env_with_state_checks():
    """Test ConveyorBeltEnv movement without wraparound (boxes fall off the
    right)."""
    env = ConveyorBeltEnv(auto_drop=False)
    video_dir = Path("videos/test-conveyorbelt-env-no-wrap")
    video_dir.mkdir(parents=True, exist_ok=True)

    wrapped_env = RecordVideo(
        env,
        str(video_dir),
        episode_trigger=lambda episode_id: episode_id == 0,
    )

    wrapped_env.action_space.seed(42)
    state, _ = wrapped_env.reset(seed=42)
    assert isinstance(state, ConveyorBeltState)

    current_positions = state.positions.copy()
    belt_length = env.scene_spec.belt_length
    box_width = env.scene_spec.box_width
    dt = env.scene_spec.dt

    speed_map = {
        0: -1.0,  # reverse
        1: 0.0,  # stop
        2: 0.5,  # slow
        3: 1.0,  # normal
        4: 1.5,  # fast
    }

    action_hold_steps = 10
    total_steps = 200

    current_action = wrapped_env.action_space.sample()

    for step in range(total_steps):
        if step % action_hold_steps == 0:
            current_action = wrapped_env.action_space.sample()

        assert isinstance(current_action, ConveyorBeltAction)
        action_index = current_action.index

        obs, _, terminated, truncated, _ = wrapped_env.step(current_action)
        assert isinstance(obs, ConveyorBeltState)

        expected_velocity = speed_map.get(action_index)
        np.testing.assert_array_almost_equal(
            obs.velocities,
            expected_velocity,
            err_msg=f"Velocity mismatch on action {action_index}",
        )

        # new expected: linear advance, then removal of anything after end
        updated = current_positions + expected_velocity * dt
        updated = updated[updated < (belt_length - box_width)]

        np.testing.assert_array_almost_equal(
            np.sort(obs.positions),
            np.sort(updated),
            err_msg=f"Position update mismatch on action {action_index}",
        )

        # Move forward
        current_positions = obs.positions.copy()

        assert not terminated and not truncated, "Episode ended prematurely"

    wrapped_env.close()


def test_automatic_box_dropping_behavior():
    """Test auto box drop logic using held actions, no wrap, with spacing
    constraints."""
    env = ConveyorBeltEnv(auto_drop=True)
    video_dir = Path("videos/test-conveyorbelt-env-auto")
    video_dir.mkdir(parents=True, exist_ok=True)

    wrapped_env = RecordVideo(
        env,
        str(video_dir),
        name_prefix="auto-drop",
        episode_trigger=lambda episode_id: episode_id == 0,
    )

    wrapped_env.action_space.seed(42)
    state, _ = wrapped_env.reset(seed=42)

    initial_count = len(state.positions)
    min_spacing = env.scene_spec.min_spacing

    assert initial_count > 0, "Environment should start with boxes"

    speed_map = {0: -1.0, 1: 0.0, 2: 0.5, 3: 1.0, 4: 1.5}
    action_hold_steps = 10
    total_steps = 200

    current_action = wrapped_env.action_space.sample()
    prev_count = initial_count
    drop_frames = []

    for step in range(total_steps):
        if step % action_hold_steps == 0:
            current_action = wrapped_env.action_space.sample()

        assert isinstance(current_action, ConveyorBeltAction)
        action_index = current_action.index

        state, _, terminated, truncated, _ = wrapped_env.step(current_action)

        expected_v = speed_map[action_index]
        np.testing.assert_array_almost_equal(
            state.velocities,
            expected_v,
            err_msg=f"Velocity mismatch on action {action_index}",
        )

        # enforce spacing check (only boxes currently on belt, i.e., not falling)
        sorted_on_belt = [
            p for p, h in zip(state.positions, state.falling_heights) if h == 0
        ]
        sorted_on_belt = np.sort(sorted_on_belt)
        if len(sorted_on_belt) > 1:
            diffs = np.diff(sorted_on_belt)
            tolerance = 0.02
            assert all(
                d + tolerance >= min_spacing for d in diffs
            ), f"Overlap at step {step}: diffs={diffs}"
        this_count = len(state.positions)
        if this_count > prev_count:
            drop_frames.append(step)
        prev_count = this_count

        assert not terminated and not truncated

    # should have auto-dropped at least one box
    assert drop_frames, "No boxes auto-dropped with auto_drop=True"

    # drops should not be overly frequent
    if len(drop_frames) > 1:
        deltas = np.diff(drop_frames)
        assert all(d > 1 for d in deltas), f"Drops too frequent: {deltas}"

    wrapped_env.close()
