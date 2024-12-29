"""A controller for the hovercraft environment."""

import control as ct
import numpy as np

from hybrid_failure_discovery.envs.hovercraft_env import (
    HoverCraftAction,
    HoverCraftSceneSpec,
    HoverCraftState,
)


class HoverCraftParameterizedController:
    """An LQR-based controller that takes in a single boolean parameter that
    indicates whether the hovercraft should switch between goal pairs."""

    def __init__(self, scene_spec: HoverCraftSceneSpec) -> None:
        self._scene_spec = scene_spec

        # Prepare LQR.
        A = self._scene_spec.A
        B = self._scene_spec.B
        Q = self._scene_spec.Q
        R = self._scene_spec.R
        self._K, _, _ = ct.dlqr(A, B, Q, R)

        self._goal_pair_index: tuple[int, int] = (0, 0)

    def reset(self, obs: HoverCraftState) -> None:
        """Reset the controller."""
        self._goal_pair_index = self._scene_spec.get_goal_pair_index_from_state(obs)

    def step(self, obs: HoverCraftState, high_level_action: bool) -> HoverCraftAction:
        """Optionally toggle the goal pair and then return an LQR action."""

        # Change goal to match observation.
        if high_level_action:
            self._goal_pair_index = self._scene_spec.get_goal_pair_index_from_state(obs)

        # Get LQR action.
        state_vec = np.array([obs.x, obs.vx, obs.y, obs.vy])
        gi, gj = self._goal_pair_index
        gx, gy = self._scene_spec.goal_pairs[gi][gj]
        goal_vec = np.array([gx, 0, gy, 0])
        error_vec = np.subtract(state_vec, goal_vec)
        action_vec = -self._K @ error_vec
        ux, uy = action_vec
        return HoverCraftAction(ux, uy)
