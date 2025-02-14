"""A controller for the hovercraft environment."""

import control as ct
import numpy as np
from gymnasium.spaces import Space
from tomsutils.spaces import EnumSpace

from hybrid_failure_discovery.controllers.controller import ConstraintBasedController
from hybrid_failure_discovery.envs.hovercraft_env import (
    HoverCraftAction,
    HoverCraftSceneSpec,
    HoverCraftState,
)


class HoverCraftController(
    ConstraintBasedController[HoverCraftState, HoverCraftAction]
):
    """An LQR-based controller that is parameterized by the amount of time to
    delay before switching between up/down and left/right."""

    def __init__(
        self,
        seed: int,
        scene_spec: HoverCraftSceneSpec,
        time_delay_parameter: float = 0.05,
    ) -> None:
        super().__init__(seed)
        self._scene_spec = scene_spec
        self._time_delay_parameter = time_delay_parameter  # in seconds

        # Prepare LQR.
        A = self._scene_spec.A
        B = self._scene_spec.B
        Q = self._scene_spec.Q
        R = self._scene_spec.R
        self._K, _, _ = ct.dlqr(A, B, Q, R)

        self._goal_pair_index: tuple[int, int] = (0, 0)
        self._time_since_switch: float = 0.0

    def reset(self, initial_state: HoverCraftState) -> None:
        """Reset the controller."""
        self._goal_pair_index = self._scene_spec.get_goal_pair_index_from_state(
            initial_state
        )
        self._time_since_switch = 0.0

    def step_action_space(self, state: HoverCraftState) -> Space[HoverCraftAction]:
        """Optionally toggle the goal pair and then return an LQR action."""

        # Check if goal pair switched.
        goal_pair_index = self._scene_spec.get_goal_pair_index_from_state(state)

        # Switch.
        if self._time_since_switch >= self._time_delay_parameter:
            self._goal_pair_index = goal_pair_index
            self._time_since_switch = 0

        # Get LQR action.
        state_vec = np.array([state.x, state.vx, state.y, state.vy])
        gi, gj = self._goal_pair_index
        gx, gy = self._scene_spec.goal_pairs[gi][gj]
        goal_vec = np.array([gx, 0, gy, 0])
        error_vec = np.subtract(state_vec, goal_vec)
        action_vec = -self._K @ error_vec
        ux, uy = action_vec
        action = HoverCraftAction(ux, uy)

        self._time_since_switch += self._scene_spec.dt

        # Singleton space right now.
        action_space = EnumSpace([action])
        return action_space
