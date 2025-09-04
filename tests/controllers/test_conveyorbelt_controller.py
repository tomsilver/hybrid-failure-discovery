"""Tests for conveyorbelt_controller.py."""

from gymnasium.wrappers import RecordVideo

from hybrid_failure_discovery.controllers.conveyorbelt_controller import (
    ConveyorBeltCommand,
    ConveyorBeltController,
)
from hybrid_failure_discovery.envs.conveyorbelt_env import (
    ConveyorBeltEnv,
)


def test_conveyorbelt_controller():
    """Tests for conveyorbelt_controller.py."""

    env = ConveyorBeltEnv()
    controller = ConveyorBeltController(123, env.scene_spec)

    env = RecordVideo(env, "videos/test-conveyorbelt-controller")

    state, _ = env.reset(seed=123)
    controller.reset(state)

    for t in range(250):
        # Alternate between speed commands for testing.
        if t < 50:
            command = ConveyorBeltCommand(target_speed=0.0)  # stop
        elif t < 100:
            command = ConveyorBeltCommand(target_speed=0.5)  # slow
        elif t < 150:
            command = ConveyorBeltCommand(target_speed=1.0)  # normal
        elif t < 200:
            command = ConveyorBeltCommand(target_speed=1.5)  # fast
        else:
            command = ConveyorBeltCommand(maintain_spacing=True)  # feedback mode

        action = controller.step(state, command)
        state, _, _, _, _ = env.step(action)

    env.close()
