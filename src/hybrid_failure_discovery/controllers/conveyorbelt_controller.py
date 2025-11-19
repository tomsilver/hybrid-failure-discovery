# """A controller for the conveyorbelt environment that turns external commands
# into drop/no-drop actions for the ConveyorBeltEnv."""

# from dataclasses import dataclass
# from typing import Optional

# from gymnasium.spaces import Space
# from tomsutils.spaces import EnumSpace

# from hybrid_failure_discovery.controllers.controller import ConstraintBasedController
# from hybrid_failure_discovery.envs.conveyorbelt_env import (
#     ConveyorBeltAction,
#     ConveyorBeltSceneSpec,
#     ConveyorBeltState,
# )


# @dataclass(frozen=True)
# class ConveyorBeltCommand:
#     """External command for the conveyor belt controller.

#     Set `drop_now=True` to request a package drop on this step; False
#     means do not drop. If None, defaults to False (no drop).
#     """

#     drop_now: Optional[bool] = None


# class ConveyorBeltController(
#     ConstraintBasedController[
#         ConveyorBeltState, ConveyorBeltAction, ConveyorBeltCommand
#     ]
# ):
#     """A simple pass-through controller: command -> ConveyorBeltAction."""

#     def __init__(self, seed: int, scene_spec: ConveyorBeltSceneSpec) -> None:
#         super().__init__(seed)
#         self._scene_spec = scene_spec
#         self._initial_state: Optional[ConveyorBeltState] = None

#     def reset(self, initial_state: ConveyorBeltState) -> None:
#         """Store initial state (no internal dynamics needed)."""
#         self._initial_state = initial_state

#     def step_action_space(
#         self, state: ConveyorBeltState, command: ConveyorBeltCommand
#     ) -> Space[ConveyorBeltAction]:
#         """Translate the external command into an action for this step.

#         If `command.drop_now` is True -> drop a package.
#         If False or None -> do not drop.
#         """
#         drop = bool(command.drop_now)  # None -> False
#         action = ConveyorBeltAction(drop_package=drop)
#         return EnumSpace([action])

#     def get_command_space(self) -> Space[ConveyorBeltCommand]:
#         """Enumerate representative commands (no-drop / drop)."""
#         return EnumSpace(
#             [
#                 ConveyorBeltCommand(drop_now=False),
#                 ConveyorBeltCommand(drop_now=True),
#             ]
#         )
# """A controller for the conveyorbelt environment that implements
# mode-based auto-dropping (off/slow/mid/fast) plus a buggy spacing
# "safety" check based only on time since last drop.
# """

# from dataclasses import dataclass
# from typing import Optional

# from gymnasium.spaces import Space
# from tomsutils.spaces import EnumSpace

# from hybrid_failure_discovery.controllers.controller import ConstraintBasedController
# from hybrid_failure_discovery.envs.conveyorbelt_env import (
#     ConveyorBeltAction,
#     ConveyorBeltSceneSpec,
#     ConveyorBeltState,
# )


# @dataclass(frozen=True)
# class ConveyorBeltCommand:
#     """High-level command for the conveyor belt controller.

#     mode:
#         - "off"   : never drop automatically
#         - "slow"  : drop rarely
#         - "mid"   : medium rate
#         - "fast"  : drop frequently
#     """

#     mode: str = "off"  # one of {"off", "slow", "mid", "fast"}


# class ConveyorBeltController(
#     ConstraintBasedController[
#         ConveyorBeltState, ConveyorBeltAction, ConveyorBeltCommand
#     ]
# ):
#     """Auto-dropping controller with a deliberately buggy safety check.

#     - Translates a high-level rate mode ("off"/"slow"/"mid"/"fast")
#       into a sequence of drop/no-drop actions.
#     - Uses ONLY time since last drop (not actual positions) to decide
#       if it's "safe" to drop, and does so in a buggy way on purpose.
#     """

#     def __init__(self, seed: int, scene_spec: ConveyorBeltSceneSpec) -> None:
#         super().__init__(seed)
#         self._scene_spec = scene_spec
#         self._initial_state: Optional[ConveyorBeltState] = None

#         # Number of steps between drops for each mode.
#         # Smaller number => higher drop rate.
#         self._mode_to_period_steps = {
#             "off": None,  # no auto dropping
#             "slow": 150,
#             "mid": 25,
#             "fast": 8,
#         }

#         # Internal counters for auto-dropping/safety logic
#         self._steps_since_last_drop: int = 10**9  # start very large so first drop is allowed
#         self._last_drop_step: Optional[int] = None

#     def reset(self, initial_state: ConveyorBeltState) -> None:
#         """Reset the controller to initial state."""
#         self._initial_state = initial_state
#         self._steps_since_last_drop = 10**9
#         self._last_drop_step = None

#     def step_action_space(
#         self, state: ConveyorBeltState, command: ConveyorBeltCommand
#     ) -> Space[ConveyorBeltAction]:
#         """Compute the next action space given the current state and command.

#         Implements:
#         - Mode-based auto-dropping:
#             * "off"  -> never drop
#             * "slow" -> drop every ~60 steps
#             * "mid"  -> drop every ~25 steps
#             * "fast" -> drop every ~8 steps
#         - A buggy "safety" check that uses only time since last drop
#           to approximate the last box's x-position.
#         """

#         # Advance "time since last drop" counter
#         self._steps_since_last_drop += 1

#         # Base rate decision from mode
#         period_steps = self._mode_to_period_steps.get(command.mode, None)
#         if period_steps is None:
#             # "off" or unknown mode -> never drop
#             want_drop = False
#         else:
#             want_drop = self._steps_since_last_drop >= period_steps

#         drop = want_drop

#         # ------------- BUGGY SAFETY CHECK ------------------------------------
#         # Intention:
#         #   Don't drop if the last box hasn't moved far enough along x.
#         #
#         # Implementation (deliberately flawed):
#         #   - Estimate x-distance traveled using only time since last drop:
#         #         est_dx = v * steps_since_last_drop * dt
#         #   - Compare est_dx to min_spacing.
#         #   - If "too close", skip drop.
#         #
#         # Bugs:
#         #   1) Ignores fall time (box isn't even on belt yet).
#         #   2) Ignores actual positions and multiple boxes.
#         #   3) Resets steps_since_last_drop even when *skipping* a drop,
#         #      which keeps the estimate artificially small and can lead
#         #      to weird dynamics and missed safety enforcement later.
#         # ----------------------------------------------------------------------
#         if drop and self._last_drop_step is not None:
#             v = float(self._scene_spec.conveyor_belt_velocity)
#             dt = float(self._scene_spec.dt)
#             min_spacing = float(getattr(self._scene_spec, "min_spacing", 0.0))

#             estimated_distance = v * self._steps_since_last_drop * dt

#             # Buggy comparison: uses <=, ignores box_width, and uses estimate only
#             if estimated_distance <= min_spacing:
#                 # Decide it's "unsafe" and skip this drop
#                 drop = False
#                 # BUG: reset this even though we *didn't* drop
#                 # self._steps_since_last_drop = 0

#         # If we actually drop, reset counters
#         if drop:
#             self._steps_since_last_drop = 0
#             self._last_drop_step = 0 if self._last_drop_step is None else self._last_drop_step + 1

#         action = ConveyorBeltAction(drop_package=drop)
#         return EnumSpace([action])

#     def get_command_space(self) -> Space[ConveyorBeltCommand]:
#         """Enumerate representative high-level commands (rate modes)."""
#         return EnumSpace(
#             [
#                 ConveyorBeltCommand(mode="off"),
#                 ConveyorBeltCommand(mode="slow"),
#                 ConveyorBeltCommand(mode="mid"),
#                 ConveyorBeltCommand(mode="fast"),
#             ]
#         )

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
