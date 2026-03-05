"""Failure monitor for the hovercraft environment."""

import numpy as np
from tomsgeoms2d.structs import Circle, geom2ds_intersect

from gym_failure_discovery.envs.hovercraft_env import HoverCraftSceneSpec


class HoverCraftFailureMonitor:
    """Detects collisions between the hovercraft and obstacles."""

    def __init__(self, scene_spec: HoverCraftSceneSpec) -> None:
        self._scene_spec = scene_spec

    def reset(self, obs: np.ndarray) -> None:
        """No internal state to reset."""

    def step(  # pylint: disable=unused-argument
        self, obs: np.ndarray, action: int, next_obs: np.ndarray
    ) -> bool:
        """Return True if the hovercraft collides with an obstacle."""
        x, _, y, _ = next_obs[:4]
        circ = Circle(float(x), float(y), self._scene_spec.hovercraft_radius)
        return any(geom2ds_intersect(circ, o) for o in self._scene_spec.obstacles)
