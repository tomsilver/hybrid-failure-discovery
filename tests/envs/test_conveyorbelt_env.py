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

    env = ConveyorBeltEnv(seed=42)

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

    rng = np.random.default_rng(42)

    for step in range(10):
        # Sample random action
        action_index = int(rng.integers(low=0, high=len(wrapped_env.action_space)))
        action = ConveyorBeltAction(index=action_index)

        # Step the environment
        next_state, _, done, truncated, _ = wrapped_env.step(action)

        assert isinstance(next_state, ConveyorBeltState)

        next_values = next_state.values

        print(f"\nStep {step + 1}")
        if action_index == 0:
            print("Action taken: 0 (Do nothing)")
        elif action_index == 1:
            print("Action taken: 1 (Shift left, add block on right)")
        elif action_index == 2:
            print("Action taken: 2 (Shift right, add block on left)")
        else:
            print(f"Invalid action taken: {action_index} (Unknown action)")

        print(f"State before: {current_values}")
        if action_index in (1, 2):
            inserted_value = getattr(
                wrapped_env.unwrapped, "_last_inserted_value", None
            )
            if inserted_value is not None:
                print(f"New value inserted: {inserted_value:.4f}")
            else:
                print("New value inserted: None")
        print(f"State after:  {next_values}")

        # Compute expected next_values based on action and current_values
        expected_values = current_values.copy()
        if action_index == 0:
            # No-op: state should remain unchanged
            pass
        elif action_index == 1:
            # Shift left, new random box on right
            # The new box value is unknown, should match the last element in next_values
            expected_values[:-1] = current_values[1:]
            expected_values[-1] = next_values[-1]
        elif action_index == 2:
            # Shift right, new random box on left
            expected_values[1:] = current_values[:-1]
            expected_values[0] = next_values[0]
        else:
            pytest.fail(f"Unexpected action index: {action_index}")

        print(f"Expected state: {expected_values}")

        # Assert arrays are almost equal (allowing for floating point)

        np.testing.assert_array_almost_equal(
            next_values,
            expected_values,
            err_msg=f"State transition mismatch on action {action_index}",
        )

        # Update current_values for next iteration
        current_values = next_values.copy()

        # Handle episode end properly to ensure video is saved
        if done or truncated:
            print(f"Episode ended at step {step + 1}, resetting environment.")
            state, _ = wrapped_env.reset()
            current_values = state.values.copy()

    wrapped_env.close()
