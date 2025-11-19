# """Tests for conveyorbelt_controller.py (drop/no-drop command)."""

# from pathlib import Path

# from gymnasium.wrappers import RecordVideo

# from hybrid_failure_discovery.controllers.conveyorbelt_controller import (
#     ConveyorBeltCommand,
#     ConveyorBeltController,
# )
# from hybrid_failure_discovery.envs.conveyorbelt_env import ConveyorBeltEnv


# def test_conveyorbelt_controller_drop_commands():
#     """Drive the env with explicit drop/no-drop commands and ensure it runs
#     cleanly."""
#     # --- Env & controller setup ---
#     env = ConveyorBeltEnv()
#     controller = ConveyorBeltController(seed=123, scene_spec=env.scene_spec)

#     video_dir = Path("videos/test-conveyorbelt-controller")
#     video_dir.mkdir(parents=True, exist_ok=True)
#     env = RecordVideo(env, str(video_dir), episode_trigger=lambda eid: eid == 0)

#     state, _ = env.reset(seed=123)
#     controller.reset(state)

#     # --- Command schedule: drop at specific timesteps, otherwise don't drop ---
#     # e.g., drop at t in {0, 5, 10, 30, 31, 60, 90}
#     drop_times = {0, 5, 10, 30, 31, 60, 90}

#     for t in range(100):
#         command = ConveyorBeltCommand(drop_now=t in drop_times)
#         action = controller.step(
#             state, command
#         )  # controller returns a concrete ConveyorBeltAction
#         state, _, terminated, truncated, _ = env.step(action)
       
#         # Sanity: the test should not end early
#         assert not terminated and not truncated

#     env.close()
"""Tests for conveyorbelt_controller.py (mode-based auto-drop, varied modes)."""

from pathlib import Path

from gymnasium.wrappers import RecordVideo

from hybrid_failure_discovery.controllers.conveyorbelt_controller import (
    ConveyorBeltCommand,
    ConveyorBeltController,
)
from hybrid_failure_discovery.envs.conveyorbelt_env import ConveyorBeltEnv


def test_conveyorbelt_controller_varied_mode_schedule():
    """Exercise the controller with a varied schedule of modes (off/slow/mid/fast)
    and ensure the controller–env integration runs without errors.

    This is a smoke test: we just run the loop and print some diagnostics,
    similar in spirit to the hovercraft controller tests.
    """

    # --- Env & controller setup ---
    env = ConveyorBeltEnv()
    controller = ConveyorBeltController(seed=123, scene_spec=env.scene_spec)

    video_dir = Path("videos/test-conveyorbelt-controller")
    video_dir.mkdir(parents=True, exist_ok=True)
    env = RecordVideo(env, str(video_dir), episode_trigger=lambda eid: eid == 0)

    state, _ = env.reset(seed=123)
    controller.reset(state)

    # Varied mode schedule:
    # mix off/slow/mid/fast in a repeating pattern
    modes_cycle = [
        "off",
        "slow",
        "mid",
        "fast",
        "off",
        "slow",
        "mid",
        "off",
    ]

    total_steps = 300
    total_drops = 0

    for t in range(total_steps):
        mode = modes_cycle[t % len(modes_cycle)]
        command = ConveyorBeltCommand(mode=mode)

        # Controller turns high-level mode into concrete ConveyorBeltAction
        action = controller.step(state, command)

        if getattr(action, "drop_package", False):
            total_drops += 1

        state, _, terminated, truncated, _ = env.step(action)

        # If the env ends the episode (e.g., horizon), reset and keep going
        if terminated or truncated:
            state, _ = env.reset(seed=123)
            controller.reset(state)

    env.close()
    print(f"[conveyorbelt_controller] total steps={total_steps}, total_drops={total_drops}")
