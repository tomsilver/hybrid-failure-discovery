"""A controller for the hovercraft environment."""

from dataclasses import dataclass

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


@dataclass(frozen=True)
class HoverCraftCommand:
    """A command in the hovercraft environment."""

    switch: bool  # whether to switch between left, right and up, down flight


class HoverCraftController(
    ConstraintBasedController[HoverCraftState, HoverCraftAction, HoverCraftCommand]
):
    """An LQR-based controller for the hovercraft environment."""

    def __init__(
        self,
        seed: int,
        scene_spec: HoverCraftSceneSpec,
    ) -> None:
        super().__init__(seed)
        self._scene_spec = scene_spec

        # Prepare LQR.
        A = self._scene_spec.A
        B = self._scene_spec.B
        Q = self._scene_spec.Q
        R = self._scene_spec.R
        self._K, _, _ = ct.dlqr(A, B, Q, R)

        self._goal_pair_index = (0, 0)

    def reset(self, initial_state: HoverCraftState) -> None:
        """Reset the controller."""
        self._goal_pair_index = (0, 0)

    def step_action_space(
        self, state: HoverCraftState, command: HoverCraftCommand
    ) -> Space[HoverCraftAction]:
        """Optionally toggle the goal pair and then return an LQR action."""

        # Handle goal updates first.
        state_vec = np.array([state.x, state.vx, state.y, state.vy])
        gi, gj = self._goal_pair_index
        gx, gy = self._scene_spec.goal_pairs[gi][gj]
        goal_vec = np.array([gx, 0, gy, 0])

        # First, always toggle if reached goal.
        if np.allclose(goal_vec, state_vec, atol=self._scene_spec.goal_atol):
            gj = int(not gj)

        # Switch if commanded to.
        if command.switch:
            gi = int(not gi)
            gj = 0  # arbitrarily always start with left or down

        self._goal_pair_index = (gi, gj)
        gx, gy = self._scene_spec.goal_pairs[gi][gj]

        # Get LQR action.
        goal_vec = np.array([gx, 0, gy, 0])
        error_vec = np.subtract(state_vec, goal_vec)
        action_vec = -self._K @ error_vec
        ux, uy = action_vec
        action = HoverCraftAction(ux, uy)

        # Singleton space right now.
        action_space = EnumSpace([action])
        return action_space

    def get_command_space(self) -> Space[HoverCraftCommand]:
        return EnumSpace([HoverCraftCommand(True), HoverCraftCommand(False)])
