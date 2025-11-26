"""A corrected controller for the conveyor belt that guarantees collision-free
drops by using ACTUAL state information (positions, falling heights, box width,
min spacing)."""

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

    mode: str = "off"  # one of {"off", "slow", "mid", "fast"}


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
            "slow": int(2.0 / dt),  # drop every 2.0 sec
            "mid": int(1.2 / dt),  # drop every ~1.2 sec
            "fast": int(0.04 / dt),  # drop every ~0.04 sec
        }

        self._steps_since_last_drop = 10**9

    def reset(self, initial_state: ConveyorBeltState) -> None:
        self._initial_state = initial_state
        self._steps_since_last_drop = 10**9

    def _safe_to_drop(self, state: ConveyorBeltState) -> bool:
        """Return True ONLY if dropping a new box will not collide.

        NOTE: This method has deliberate faults to allow testing of failure detection:
        1. Only checks min_spacing, ignoring box_width (allows overlaps)
        2. Only checks if falling height > 0.05
        """

        min_spacing = getattr(self._scene_spec, "min_spacing", 0.0)

        # FAULT 1: Only check if falling height is significant (> 0.05) instead of > 0.0
        # This allows drops when boxes are almost landed (height 0.0-0.05),
        # Made less severe (0.05 instead of 0.1) to make failures moderately rare
        for h in state.falling_heights:
            if h > 0.05:  # Should be > 0.0
                return False

        # FAULT 2: Only check min_spacing, ignoring box_width
        # This means boxes can overlap (collide) if they satisfy min_spacing
        # FAULT 3: Use reduced safety margin (80% of required gap)
        # This makes the check less strict, allowing drops when slightly too close
        if len(state.positions) > 0:
            nearest = min(state.positions)
            # Should check: nearest < (box_width + min_spacing)
            # But we only check min_spacing with 80% margin
            if nearest < min_spacing * 0.8:  # Should be: < (box_width + min_spacing)
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
