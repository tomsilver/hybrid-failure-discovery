"""Failure monitor for the hovercraft environment."""

from tomsgeoms2d.structs import Circle, geom2ds_intersect

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
