"""Tests for conveyorbelt_controller.py."""

from gymnasium.wrappers import RecordVideo

from hybrid_failure_discovery.controllers.conveyorbelt_controller import (
    ConveyorBeltCommand,
    ConveyorBeltController,
)
from hybrid_failure_discovery.envs.conveyorbelt_env import (
    ConveyorBeltEnv,
    ConveyorBeltSceneSpec
)


def test_conveyorbelt_controller():
    """Tests for conveyorbelt_controller.py."""

    env = ConveyorBeltEnv(scene_spec=ConveyorBeltSceneSpec(init_positions=[0.1, 0.51, 0.92, 1.33, 1.74, 2.15]))

    controller = ConveyorBeltController(123, env.scene_spec)

    env = RecordVideo(env, "videos/test-conveyorbelt-controller")

    state, _ = env.reset(seed=123)
    controller.reset(state)

    for t in range(100):
        # Alternate between speed commands for testing.
        if t < 40:
            command = ConveyorBeltCommand(target_speed=-1)  # stop
        elif t > 41 and t < 45:
            command = ConveyorBeltCommand(target_speed=1.5)
        elif t > 45:
            command = ConveyorBeltCommand(target_speed=-1)
        else:  
            command = ConveyorBeltCommand(target_speed=1.5)
        # elif t < 100:
        #     command = ConveyorBeltCommand(target_speed=0.5)  # slow
        # elif t < 150:
        #     command = ConveyorBeltCommand(target_speed=1.0)  # normal
        # elif t < 200:
        #     command = ConveyorBeltCommand(target_speed=1.5)  # fast
        # else:
        #     command = ConveyorBeltCommand(maintain_spacing=True)  # feedback mode

        action = controller.step(state, command)
        state, _, _, _, _ = env.step(action)

    env.close()
