"""Collision/spacing failure monitor for the conveyor belt environment."""

import numpy as np

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
    - The world has exploded (state.exploded)
    """

    def __init__(self, scene_spec: ConveyorBeltSceneSpec) -> None:
        super().__init__(self._check_failures)
        self._scene_spec = scene_spec

    def _check_failures(self, state: ConveyorBeltState) -> bool:
        """Return True if the world has exploded. """
    
        # Check if world has exploded
        if state.exploded:
            return True
        
    

    def get_robustness_score(self, state: ConveyorBeltState) -> float:
        raise NotImplementedError
