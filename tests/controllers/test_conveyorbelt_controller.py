"""Tests for conveyorbelt_controller.py (mode-based auto-drop, varied
modes)."""

from pathlib import Path

from gymnasium.wrappers import RecordVideo

from hybrid_failure_discovery.controllers.conveyorbelt_controller import (
    ConveyorBeltCommand,
    ConveyorBeltController,
)
from hybrid_failure_discovery.envs.conveyorbelt_env import ConveyorBeltEnv


def test_conveyorbelt_controller_varied_mode_schedule():
    """Exercise the controller with a varied schedule of modes
    (off/slow/mid/fast) and ensure the controller–env integration runs without
    errors.

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
        "fast",
    ]

    # total_steps =
    steps_per_mode = 100
    total_drops = 0
    total_steps = 0

    for mode in modes_cycle:
        # mode = modes_cycle[t % len(modes_cycle)]
        command = ConveyorBeltCommand(mode=mode)
        print(f"mode={mode}")

        for _ in range(steps_per_mode):
            # Controller turns high-level mode into concrete ConveyorBeltAction
            action = controller.step(state, command)

            if getattr(action, "drop_package", False):
                total_drops += 1

            state, _, terminated, truncated, _ = env.step(action)

            total_steps += 1

        # If the env ends the episode (e.g., horizon), reset and keep going
        if terminated or truncated:
            print("TERMINATING!")
            state, _ = env.reset(seed=123)
            controller.reset(state)

    env.close()
    print(
        f"[conveyorbelt_controller] total steps={total_steps}, total_drops={total_drops}"
    )
