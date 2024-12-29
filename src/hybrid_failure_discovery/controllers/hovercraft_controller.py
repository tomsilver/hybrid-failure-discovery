"""A controller for the hovercraft environment."""

from tomsutils.gym_agent import Agent
from hybrid_failure_discovery.envs.hovercraft_env import HoverCraftState, HoverCraftAction, HoverCraftSceneSpec
from numpy.typing import NDArray
import numpy as np


class HoverCraftController(Agent[HoverCraftState, HoverCraftAction]):
    """A controller for the hovercraft environment."""

    def __init__(self, scene_spec: HoverCraftSceneSpec, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._scene_spec = scene_spec

        # Solve LQR for each goal.
        self._goal_to_K: dict[tuple[float, float], NDArray] = {}
        for goal_pair in self._scene_spec.goal_pairs:
            for goal in goal_pair:
                self._goal_to_K[goal] = self._solve_lqr(goal)

    def _get_action(self) -> HoverCraftAction:
        import ipdb; ipdb.set_trace()

    def _solve_lqr(self, goal: tuple[float, float]) -> NDArray:

        dt = self._scene_spec.dt

        A = np.array([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1],
        ])

        B = np.array([
            [dt]
        ])

        import ipdb; ipdb.set_trace()

