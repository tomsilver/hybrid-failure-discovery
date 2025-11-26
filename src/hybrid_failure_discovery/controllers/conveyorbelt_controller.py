"""
A corrected controller for the conveyor belt that guarantees collision-free
drops by using ACTUAL state information (positions, falling heights,
box width, min spacing).
"""

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
    """High-level drop-rate mode.

    Modes:
      - off  : never drop
      - slow : drop every ~2.0 seconds
      - mid  : drop every ~1.2 seconds
      - fast : drop every ~0.7 seconds
    """

    mode: str = "off"   # one of {"off", "slow", "mid", "fast"}


class ConveyorBeltController(
    ConstraintBasedController[
        ConveyorBeltState, ConveyorBeltAction, ConveyorBeltCommand
    ]
):
    """Collision-free auto-dropping controller.

    Fixes:
      - Uses ACTUAL state, not fake estimates
      - Ensures falling boxes cannot be collided with
      - Ensures landed boxes are spaced by ≥ box_width + min_spacing
      - Mode-based timing is now safe and realistic
    """

    def __init__(self, seed: int, scene_spec: ConveyorBeltSceneSpec) -> None:
        super().__init__(seed)
        self._scene_spec = scene_spec
        self._initial_state: Optional[ConveyorBeltState] = None

        # Convert desired timing into steps
        dt = scene_spec.dt
        self._mode_to_steps = {
            "off": None,
            "slow": int(2.0 / dt),     # drop every 2.0 sec
            "mid": int(1.2 / dt),      # drop every ~1.2 sec
            "fast": int(0.7 / dt),     # drop every ~0.7 sec
        }

        self._steps_since_last_drop = 10**9

    def reset(self, initial_state: ConveyorBeltState) -> None:
        self._initial_state = initial_state
        self._steps_since_last_drop = 10**9

    def _safe_to_drop(self, state: ConveyorBeltState) -> bool:
        """Return True ONLY if dropping a new box will not collide."""

        box_width = self._scene_spec.box_width
        min_spacing = getattr(self._scene_spec, "min_spacing", 0.0)
        required_gap = box_width + min_spacing

        # 1. If any existing box is still falling → NEVER DROP
        for h in state.falling_heights:
            if h > 0.0:
                return False

        # 2. Look at the closest box to the drop position (x=0)
        if len(state.positions) > 0:
            nearest = min(state.positions)
            if nearest < required_gap:
                return False

        return True

    def step_action_space(
        self, state: ConveyorBeltState, command: ConveyorBeltCommand
    ) -> Space[ConveyorBeltAction]:

        self._steps_since_last_drop += 1

        # Determine timing requirement for the chosen mode
        steps_required = self._mode_to_steps.get(command.mode, None)

        if steps_required is None:
            # Mode = off
            drop = False

        else:
            # Mode = slow/mid/fast
            if self._steps_since_last_drop >= steps_required:
                # Timing satisfied, now check safety
                drop = self._safe_to_drop(state)
            else:
                drop = False

        # Reset timer if we actually drop
        if drop:
            self._steps_since_last_drop = 0

        action = ConveyorBeltAction(drop_package=drop)
        return EnumSpace([action])

    def get_command_space(self) -> Space[ConveyorBeltCommand]:
        return EnumSpace(
            [
                ConveyorBeltCommand(mode="off"),
                ConveyorBeltCommand(mode="slow"),
                ConveyorBeltCommand(mode="mid"),
                ConveyorBeltCommand(mode="fast"),
            ]
        )
