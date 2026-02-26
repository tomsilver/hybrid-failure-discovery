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
      - fast : drop every ~0.04 seconds
    """

    mode: str = "off"  # one of {"off", "slow", "mid", "fast"}


class ConveyorBeltController(
    ConstraintBasedController[
        ConveyorBeltState, ConveyorBeltAction, ConveyorBeltCommand
    ]
):
    """Collision-free auto-dropping controller.

    This controller has a "secret_failure_mode_sequence" which triggers
    a failure whenever that sequence of modes is given in that exact
    order.
    """

    def __init__(
        self,
        seed: int,
        scene_spec: ConveyorBeltSceneSpec,
        secret_failure_mode_sequence: list[str],
    ) -> None:
        super().__init__(seed)
        self._scene_spec = scene_spec
        self._secret_failure_mode_sequence = secret_failure_mode_sequence
        self._initial_state: Optional[ConveyorBeltState] = None

        # Convert desired timing into steps
        dt = scene_spec.dt
        self._mode_to_steps = {
            "off": None,
            "slow": int(2.0 / dt),  # drop every 2.0 sec
            "mid": int(1.2 / dt),  # drop every ~1.2 sec
            "fast": int(0.04 / dt),  # drop every ~0.04 sec
        }
        # Check that the modes in the secret failure mode sequence are valid.
        for mode in secret_failure_mode_sequence:
            assert mode in self._mode_to_steps, f"Invalid mode: {mode}"

        self._steps_since_last_drop = 10**9
        self._current_mode_sequence: list[str] = []

    def reset(self, initial_state: ConveyorBeltState) -> None:
        self._initial_state = initial_state
        self._steps_since_last_drop = 10**9
        self._current_mode_sequence = []

    def _safe_to_drop(self, state: ConveyorBeltState) -> bool:
        """Return True ONLY if dropping a new box will not collide.

        NOTE: This method has deliberate faults to allow testing of
        failure detection:
        1. Only checks if falling height > 0.5 (allows drops when boxes
           are almost landed)
        2. NO spacing check - relies entirely on timing to allow
           collisions for testing
        """

        # FAULT: Only prevent drops when boxes are falling HIGH (> 0.5)
        # instead of > 0.0
        for h in state.falling_heights:
            if h > 0.5:
                return False

        return True

    def step_action_space(
        self, state: ConveyorBeltState, command: ConveyorBeltCommand
    ) -> Space[ConveyorBeltAction]:

        # Get the newly commanded mode
        mode = command.mode
        self._current_mode_sequence.append(mode)

        # Check if the current mode sequence is the secret one, and trigger a failure
        # if it is.
        secret_len = len(self._secret_failure_mode_sequence)
        if (
            self._current_mode_sequence[-secret_len:]
            == self._secret_failure_mode_sequence
        ):
            # KABOOM!
            explode_action = ConveyorBeltAction(drop_package=False, explode=True)
            return EnumSpace([explode_action])

        self._steps_since_last_drop += 1

        # Determine timing requirement for the chosen mode
        steps_required = self._mode_to_steps[mode]

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
