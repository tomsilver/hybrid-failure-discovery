"""Failure monitor for the hovercraft environment."""

import numpy as np
from tomsgeoms2d.structs import Circle, Geom2D, Rectangle, geom2ds_intersect

from hybrid_failure_discovery.controllers.hovercraft_controller import HoverCraftCommand
from hybrid_failure_discovery.envs.hovercraft_env import (
    HoverCraftAction,
    HoverCraftSceneSpec,
    HoverCraftState,
)
from hybrid_failure_discovery.failure_monitors.failure_monitor import (
    MemorylessStateFailureMonitor,
)

from pdb import set_trace as st

class HoverCraftFailureMonitor(
    MemorylessStateFailureMonitor[HoverCraftState, HoverCraftAction, HoverCraftCommand]
):
    """Monitors a trajectory for failures."""

    def __init__(self, scene_spec: HoverCraftSceneSpec) -> None:
        super().__init__(self._check_collisions)
        self._scene_spec = scene_spec

    def _check_collisions(self, state: HoverCraftState) -> bool:
        circ = Circle(state.x, state.y, self._scene_spec.hovercraft_radius)
        for obstacle in self._scene_spec.obstacles:
            if geom2ds_intersect(circ, obstacle):
                return True
        return False

    def _closest_distance(self, obstacle: Geom2D, hc_circ: Circle) -> float:
        """Calculate the closest distance between a geometry and a circle.

        Args:
            obstacle: The obstacle geometry
            hc_circ: The hovercraft circle

        Returns:
            The minimum distance between the obstacle and hovercraft
        """
        if isinstance(obstacle, Rectangle):
            # There is collision if: i) center of circle is in the rectangle or 
            # ii) if an edge of the rectangle is in the circle. This logic would only
            # work for rectangles that are axes-aligned. Needs to be updated for non
            # axes-aligned rectangles.
            
            xmin = obstacle.x
            xmax = obstacle.x + obstacle.width
            ymin = obstacle.y
            ymax = obstacle.y + obstacle.height
            xP = max(xmin, min(hc_circ.x, xmax))
            yP = max(ymin, min(hc_circ.y, ymax))
            
            # if xP == hc_circ.x and yP == hc_circ.y, circle center is inside rectangle
            dP = float(np.linalg.norm([xP - hc_circ.x, yP - hc_circ.y]))
            return max(0.0, dP - hc_circ.radius)
        else:
            raise NotImplementedError

    def get_robustness_score(self, state: HoverCraftState) -> float:
        """The robustness score for the hovercraft is the distance to the
        nearest obstacle.

        Args:
            state: The current state of the hovercraft.

        Returns:
            The distance to the nearest obstacle.
        """
        # Hovercraft geometry

        hc_circ = Circle(state.x, state.y, self._scene_spec.hovercraft_radius)
        distance_to_obstacles = [
            self._closest_distance(obstacle, hc_circ)
            for obstacle in self._scene_spec.obstacles
        ]
        score = min(distance_to_obstacles)
        return score
