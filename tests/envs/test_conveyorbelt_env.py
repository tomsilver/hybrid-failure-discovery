"""Unit tests for the ConveyorBelt environment."""

from pathlib import Path

from gymnasium.wrappers import RecordVideo

from hybrid_failure_discovery.envs.conveyorbelt_env import (
    ConveyorBeltAction,
    ConveyorBeltEnv,
    ConveyorBeltState,
)


def test_conveyorbelt_env_with_state_checks():
    """Test ConveyorBeltEnvExternalDrop movement with box drops at a regular
    interval."""
    env = ConveyorBeltEnv()
    video_dir = Path("videos/test-conveyorbelt-env")
    video_dir.mkdir(parents=True, exist_ok=True)

    wrapped_env = RecordVideo(
        env,
        str(video_dir),
        episode_trigger=lambda episode_id: episode_id == 0,
    )

    wrapped_env.action_space.seed(42)
    state, _ = wrapped_env.reset(seed=42)
    assert isinstance(state, ConveyorBeltState)

    # Make a sequence of actions that drops boxes at a regular interval.
    action_bools = (
        [True]
        + ([False] * 10)
        + [True]
        + ([False] * 50)
        + [True]
        + ([False] * 40)
        + [True]
        + ([False] * 70)
    )

    for action_bool in action_bools:
        action = ConveyorBeltAction(action_bool)

        obs, _, terminated, truncated, _ = wrapped_env.step(action)
        assert isinstance(obs, ConveyorBeltState)

        # expected_velocity = speed_map.get(action_index)
        # np.testing.assert_array_almost_equal(
        #     obs.velocities,
        #     expected_velocity,
        #     err_msg=f"Velocity mismatch on action {action_index}",
        # )

        # updated = current_positions + expected_velocity * dt
        # updated = updated[updated < (belt_length - box_width)]

        # np.testing.assert_array_almost_equal(
        #     np.sort(obs.positions),
        #     np.sort(updated),
        #     err_msg=f"Position update mismatch on action {action_index}",
        # )

        # current_positions = obs.positions.copy()
        assert not terminated and not truncated

    wrapped_env.close()
