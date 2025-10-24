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
    - Two boxes overlap in their x positions (Δx < box_width)
    - Boxes are too close together (Δx < min_spacing)
    """

    def __init__(self, scene_spec: ConveyorBeltSceneSpec) -> None:
        super().__init__(self._check_failures)
        self._scene_spec = scene_spec

    def _check_failures(self, state: ConveyorBeltState) -> bool:
        """Return True if any pair of boxes violates spacing or overlap."""
        if len(state.positions) <= 1:
            return False  # can't fail with one or zero boxes

        sorted_positions = np.sort(state.positions)
        diffs = np.diff(sorted_positions)

        # Minimum spacing and width thresholds
        min_spacing = getattr(self._scene_spec, "min_spacing", 0.0)
        box_width = self._scene_spec.box_width

        # Check if any two boxes are too close or overlapping
        if np.any(diffs < min_spacing) or np.any(diffs < box_width):
            return True

        return False

    def get_robustness_score(self, state: ConveyorBeltState) -> float:
        """Robustness = smallest spacing margin relative to thresholds.

        - Positive: safe margin to closest violation
        - Zero or negative: already failed
        """
        if len(state.positions) <= 1:
            return float("inf")

        sorted_positions = np.sort(state.positions)
        diffs = np.diff(sorted_positions)

        min_gap = np.min(diffs)
        min_spacing = getattr(self._scene_spec, "min_spacing", 0.0)
        box_width = self._scene_spec.box_width

        # Compute smallest margin from either threshold
        spacing_margin = min_gap - min_spacing
        overlap_margin = min_gap - box_width

        return min(spacing_margin, overlap_margin)
