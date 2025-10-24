"""A controller for the conveyorbelt environment that turns external commands
into drop/no-drop actions for the ConveyorBeltEnv."""

from dataclasses import dataclass
from typing import Optional

from gymnasium.spaces import Space
from tomsutils.spaces import EnumSpace

from hybrid_failure_discovery.controllers.controller import ConstraintBasedController
from hybrid_failure_discovery.envs.conveyorbelt_env import (
    ConveyorBeltAction,
    ConveyorBeltSceneSpec,
    ConveyorBeltState,
)


@dataclass(frozen=True)
class ConveyorBeltCommand:
    """External command for the conveyor belt controller.

    Set `drop_now=True` to request a package drop on this step; False
    means do not drop. If None, defaults to False (no drop).
    """

    drop_now: Optional[bool] = None


class ConveyorBeltController(
    ConstraintBasedController[
        ConveyorBeltState, ConveyorBeltAction, ConveyorBeltCommand
    ]
):
    """A simple pass-through controller: command -> ConveyorBeltAction."""

    def __init__(self, seed: int, scene_spec: ConveyorBeltSceneSpec) -> None:
        super().__init__(seed)
        self._scene_spec = scene_spec
        self._initial_state: Optional[ConveyorBeltState] = None

    def reset(self, initial_state: ConveyorBeltState) -> None:
        """Store initial state (no internal dynamics needed)."""
        self._initial_state = initial_state

    def step_action_space(
        self, state: ConveyorBeltState, command: ConveyorBeltCommand
    ) -> Space[ConveyorBeltAction]:
        """Translate the external command into an action for this step.

        If `command.drop_now` is True -> drop a package.
        If False or None -> do not drop.
        """
        drop = bool(command.drop_now)  # None -> False
        action = ConveyorBeltAction(drop_package=drop)
        return EnumSpace([action])

    def get_command_space(self) -> Space[ConveyorBeltCommand]:
        """Enumerate representative commands (no-drop / drop)."""
        return EnumSpace(
            [
                ConveyorBeltCommand(drop_now=False),
                ConveyorBeltCommand(drop_now=True),
            ]
        )
