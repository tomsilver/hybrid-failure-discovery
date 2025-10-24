"""Tests for conveyorbelt_controller.py (drop/no-drop command)."""

from pathlib import Path

from gymnasium.wrappers import RecordVideo

from hybrid_failure_discovery.controllers.conveyorbelt_controller import (
    ConveyorBeltCommand,
    ConveyorBeltController,
)
from hybrid_failure_discovery.envs.conveyorbelt_env import ConveyorBeltEnv


def test_conveyorbelt_controller_drop_commands():
    """Drive the env with explicit drop/no-drop commands and ensure it runs
    cleanly."""
    # --- Env & controller setup ---
    env = ConveyorBeltEnv()
    controller = ConveyorBeltController(seed=123, scene_spec=env.scene_spec)

    video_dir = Path("videos/test-conveyorbelt-controller")
    video_dir.mkdir(parents=True, exist_ok=True)
    env = RecordVideo(env, str(video_dir), episode_trigger=lambda eid: eid == 0)

    state, _ = env.reset(seed=123)
    controller.reset(state)

    # --- Command schedule: drop at specific timesteps, otherwise don't drop ---
    # e.g., drop at t in {0, 5, 10, 30, 31, 60, 90}
    drop_times = {0, 5, 10, 30, 31, 60, 90}

    for t in range(100):
        command = ConveyorBeltCommand(drop_now=t in drop_times)
        action = controller.step(
            state, command
        )  # controller returns a concrete ConveyorBeltAction
        state, _, terminated, truncated, _ = env.step(action)

        # Sanity: the test should not end early
        assert not terminated and not truncated

    env.close()
