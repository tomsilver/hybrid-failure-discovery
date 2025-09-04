"""Failure monitor for the conveyor belt environment."""

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
    """Monitors a trajectory for failures."""

    def __init__(self, scene_spec: ConveyorBeltSceneSpec) -> None:
        super().__init__(self._check_failures)
        self._scene_spec = scene_spec

    def _check_failures(self, state: ConveyorBeltState) -> bool:
        """Check if the conveyor belt is in a failure state.

        Failure occurs if:
        - A box falls off the left side of the belt (pos < 0).
        - Boxes are too close together (< min_spacing).
        """
        # 1. Box falls off the left edge
        for pos in state.positions:
            if pos < 0.0:
                return True

        # 2. Boxes too close together
        if len(state.positions) > 1:
            sorted_positions = np.sort(state.positions)
            diffs = np.diff(sorted_positions)
            if np.any(diffs < self._scene_spec.min_spacing):
                return True

        return False

    def get_robustness_score(self, state: ConveyorBeltState) -> float:
        """Robustness = margin from failure.

        - Distance of boxes from the left edge (we ignore the right).
        - Distances between neighboring boxes.
        """
        robustness_scores = []

        # Distance to left belt edge only
        for pos in state.positions:
            robustness_scores.append(pos)  # since left edge is at 0

        # Distance between boxes
        if len(state.positions) > 1:
            sorted_positions = np.sort(state.positions)
            diffs = np.diff(sorted_positions)
            robustness_scores.extend(diffs.tolist())

        # If no scores (empty state), treat as safe
        if not robustness_scores:
            return float("inf")

        return min(robustness_scores)
