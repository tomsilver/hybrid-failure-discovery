"""Collision/spacing failure monitor for the conveyor belt environment."""

from hybrid_failure_discovery.controllers.conveyorbelt_controller import (
    ConveyorBeltCommand,
)
from hybrid_failure_discovery.envs.conveyorbelt_env import (
    ConveyorBeltAction,
    ConveyorBeltSceneSpec,
    ConveyorBeltState,
)
from hybrid_failure_discovery.failure_monitors.failure_monitor import (
    MemorylessStateFailureMonitor,
)


class ConveyorBeltFailureMonitor(
    MemorylessStateFailureMonitor[
        ConveyorBeltState, ConveyorBeltAction, ConveyorBeltCommand
    ]
):
    """Monitors for collisions or spacing violations on the conveyor belt.

    Failure occurs if:
    - Two boxes overlap in their x positions (Δx < box_width)
    - Boxes are too close together (Δx < min_spacing)
    """

    def __init__(self, scene_spec: ConveyorBeltSceneSpec) -> None:
        super().__init__(self._check_failures)
        self._scene_spec = scene_spec

    def _check_failures(self, state: ConveyorBeltState) -> bool:
        """Return True if the world has exploded.."""
        return state.exploded

    def get_robustness_score(self, state: ConveyorBeltState) -> float:
        raise NotImplementedError
