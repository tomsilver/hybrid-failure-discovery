"""Unit tests for the ConveyorBelt environment.

This module tests the functionality of the ConveyorBeltEnv class, including:
- Valid state transitions based on discrete actions (no-op, shift left, shift right)
- Proper video recording during environment interaction
- Correct insertion and shifting behavior of box values on the conveyor belt
"""

from pathlib import Path

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo

from hybrid_failure_discovery.envs.conveyorbelt_env import (
    ConveyorBeltAction,
    ConveyorBeltEnv,
    ConveyorBeltState,
)


def test_conveyorbelt_env_with_state_checks():
    """Detailed test of ConveyorBeltEnv with video recording and state
    transition checks."""

    env = ConveyorBeltEnv()
    video_dir = Path("videos/test-conveyorbelt-env")
    video_dir.mkdir(parents=True, exist_ok=True)

    wrapped_env = RecordVideo(
        env,
        str(video_dir),
        episode_trigger=lambda episode_id: episode_id == 0,
    )

    # Reset environment and get initial state
    state, _ = wrapped_env.reset(seed=42)
    assert isinstance(state, ConveyorBeltState)
    current_values = state.values.copy()
    print(f"\nInitial state values: {current_values}")

    for _ in range(10):
        # Sample random action directly (returns ConveyorBeltAction)
        action = wrapped_env.action_space.sample()
        assert isinstance(action, ConveyorBeltAction)
        action_index = action.index

        # Step the environment
        obs, _, terminated, truncated, _ = wrapped_env.step(action)
        assert isinstance(obs, ConveyorBeltState)

        next_values = obs.values
        expected_values = current_values.copy()

        if action_index == 0:
            pass  # No-op
        elif action_index == 1:
            print("Shift left, add right")
            expected_values[:-1] = current_values[1:]
            expected_values[-1] = next_values[-1]  # Preserve new value
        elif action_index == 2:
            print("Shift right, add left")
            expected_values[1:] = current_values[:-1]
            expected_values[0] = next_values[0]  # Preserve new value
        else:
            pytest.fail(f"Unexpected action index: {action_index}")

        np.testing.assert_array_almost_equal(
            next_values,
            expected_values,
            err_msg=f"State transition mismatch on action {action_index}",
        )

        current_values = next_values.copy()

        # Reset if episode ends (not expected for this env, but good practice)
        if terminated or truncated:
            state, _ = wrapped_env.reset()
            current_values = state.values.copy()

    wrapped_env.close()
