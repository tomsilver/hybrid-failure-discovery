"""A controller for the hovercraft environment."""

from tomsutils.gym_agent import Agent
from hybrid_failure_discovery.envs.hovercraft_env import HoverCraftState, HoverCraftAction, HoverCraftSceneSpec
from numpy.typing import NDArray
import control as ct
import numpy as np


class HoverCraftController(Agent[HoverCraftState, HoverCraftAction]):
    """A controller for the hovercraft environment."""

    def __init__(self, scene_spec: HoverCraftSceneSpec, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._scene_spec = scene_spec

        # Solve LQR.
        self._K = self._solve_lqr()

    def _get_action(self) -> HoverCraftAction:
        state = self._last_observation
        state_vec = np.array([state.x, state.vx, state.y, state.vy])
        gx = state.gx
        gy = state.gy
        goal_vec = np.array([gx, 0, gy, 0])
        error_vec = np.subtract(state_vec, goal_vec)
        action_vec = -self._K @ error_vec
        ux, uy = action_vec
        return HoverCraftAction(ux, uy)

    def _solve_lqr(self) -> NDArray:
        A = self._scene_spec.A
        B = self._scene_spec.B
        Q = self._scene_spec.Q
        R = self._scene_spec.R
        K, _, _ = ct.dlqr(A, B, Q, R)
        return K
