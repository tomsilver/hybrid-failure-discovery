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

    # ensures reproducability
    wrapped_env.action_space.seed(seed=42)
    # Reset environment and get initial state
    state, _ = wrapped_env.reset(seed=42)
    assert isinstance(state, ConveyorBeltState)
    current_values = state.values.copy()

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
            # Reverse: all values set to -1.0
            expected_values[:] = -1.0
        elif action_index == 1:
            # Stop: all values set to 0.0
            expected_values[:] = 0.0
        elif action_index == 2:
            # Slow Forward: all values set to 0.5
            expected_values[:] = 0.5
        elif action_index == 3:
            # Normal Speed: all values set to 1.0
            expected_values[:] = 1.0
        elif action_index == 4:
            # Fast: all values set to 1.5
            expected_values[:] = 1.5
        else:
            pytest.fail(f"Unexpected action index: {action_index}")

        np.testing.assert_array_almost_equal(
            next_values,
            expected_values,
            err_msg=f"State transition mismatch on action {action_index}",
        )

        current_values = next_values.copy()

        assert terminated or truncated is False, "End of episode, invalid reset"

    wrapped_env.close()
