"""Failure monitor for the hovercraft environment."""

from tomsgeoms2d.structs import Circle, Rectangle, geom2ds_intersect, circles_intersect, line_segment_intersects_circle
import numpy as np

from hybrid_failure_discovery.envs.hovercraft_env import (
    HoverCraftAction,
    HoverCraftSceneSpec,
    HoverCraftState,
)
from hybrid_failure_discovery.failure_monitors.failure_monitor import (
    MemorylessStateFailureMonitor,
)


class HoverCraftFailureMonitor(
    MemorylessStateFailureMonitor[HoverCraftState, HoverCraftAction]
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
    
    def closest_distance(self, obstacle: Rectangle, hc_circ: Circle) -> bool:
        """Checks if a rectangle intersects a circle."""
        # Optimization: if the circumscribed circle of the rectangle doesn't
        # intersect with the circle, then there can't be an intersection.
        xmin = obstacle.x
        xmax = obstacle.x + obstacle.width
        ymin = obstacle.y
        ymax = obstacle.y + obstacle.height
        xP = max(xmin, min(hc_circ.x, xmax))
        yP = max(ymin, min(hc_circ.y, ymax))

        # if xP == hc_circ.x and yP == hc_circ.y, circle center is inside the rectangle 
        dP = np.linalg.norm([xP-hc_circ.x, yP-hc_circ.y])
        return max(0, dP - hc_circ.radius)

    
    def get_closest_distance(self, state: HoverCraftState) -> float:
        hc_circ = Circle(state.x, state.y, self._scene_spec.hovercraft_radius)
        dist_to_obs = []
        for obstacle in self._scene_spec.obstacles:
            dist_to_obs.append(self.closest_distance(obstacle, hc_circ))
        return min(dist_to_obs)
