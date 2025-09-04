"""A controller for the conveyorbelt environment."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
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
    """Command for the conveyor belt controller."""

    target_speed: Optional[float] = None  # Fraction of max speed (-1.0 to 1.5 allowed)
    maintain_spacing: bool = False  # Use feedback to maintain desired spacing


class ConveyorBeltController(
    ConstraintBasedController[
        ConveyorBeltState, ConveyorBeltAction, ConveyorBeltCommand
    ]
):
    """A feedback-based controller for the conveyor belt environment."""

    def __init__(
        self,
        seed: int,
        scene_spec: ConveyorBeltSceneSpec,
    ) -> None:
        super().__init__(seed)
        self._scene_spec = scene_spec

        # For spacing control
        self._desired_spacing = getattr(scene_spec, "desired_spacing", None)

        # Local speed map (matches environment)
        self._speed_map = {
            0: -1.0,  # reverse
            1: 0.0,  # stop
            2: 0.5,  # slow
            3: 1.0,  # normal
            4: 1.5,  # fast
        }
        self._initial_state: Optional[ConveyorBeltState] = None

    def reset(self, initial_state: ConveyorBeltState) -> None:
        """Reset the controller to initial state:
        - Boxes spaced according to initial positions
        - All velocities set to 0"""
        # Store a reference state that matches the environment's initial spacing
        self._initial_state = ConveyorBeltState(
            positions=np.array(self._scene_spec.init_positions, dtype=np.float32),
            velocities=np.zeros_like(
                self._scene_spec.init_velocities, dtype=np.float32
            ),
            falling_heights=np.zeros(
                len(self._scene_spec.init_positions), dtype=np.float32
            ),
        )

    def step_action_space(
        self, state: ConveyorBeltState, command: ConveyorBeltCommand
    ) -> Space[ConveyorBeltAction]:
        """Compute the next action space given the current state and command.

        Supports single speed or maintain-spacing mode.
        """

        # Determine base target speed
        if command.target_speed is not None:
            speed_fraction = command.target_speed
        else:
            speed_fraction = 0.0  # default stop if nothing commanded

        # Maintain spacing if requested
        if command.maintain_spacing and self._desired_spacing is not None:
            if state.positions.size > 1:
                positions = np.sort(state.positions)
                spacing_errors = [
                    (positions[i + 1] - positions[i]) - self._desired_spacing
                    for i in range(len(positions) - 1)
                ]
                avg_error = np.mean(spacing_errors) if spacing_errors else 0.0
                correction = -0.5 * avg_error  # proportional control gain
                speed_fraction = np.clip(speed_fraction + correction, -1.0, 1.5)

        # Clip final speed fraction
        speed_fraction = np.clip(speed_fraction, -1.0, 1.5)

        # Pick the index whose speed is closest
        best_index = min(
            self._speed_map, key=lambda i: abs(self._speed_map[i] - speed_fraction)
        )
        best_action = ConveyorBeltAction(index=best_index)

        return EnumSpace([best_action])

    def get_command_space(self) -> Space[ConveyorBeltCommand]:
        """Enumerate some representative commands."""
        return EnumSpace(
            [
                ConveyorBeltCommand(target_speed=-1.0),
                ConveyorBeltCommand(target_speed=0.0),
                ConveyorBeltCommand(target_speed=0.5),
                ConveyorBeltCommand(target_speed=1.0),
                ConveyorBeltCommand(target_speed=1.5),
                ConveyorBeltCommand(maintain_spacing=True),
            ]
        )
