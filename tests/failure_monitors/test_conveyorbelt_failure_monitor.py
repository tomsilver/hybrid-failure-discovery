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
    BUT only after the first box has landed.
    """

    def __init__(self, scene_spec: ConveyorBeltSceneSpec) -> None:
        super().__init__(self._check_failures)
        self._scene_spec = scene_spec

    def _check_failures(self, state: ConveyorBeltState) -> bool:
        """Return True if any pair of landed boxes violates spacing or
        overlap."""
        # Mask only boxes that have landed
        landed_mask = state.falling_heights == 0.0
        landed_positions = state.positions[landed_mask]

        # If no box has landed yet, don't consider failures
        if len(landed_positions) == 0:
            return False

        # If only one box has landed, still no risk of failure
        if len(landed_positions) == 1:
            return False

        # Compute x-distance gaps between consecutive landed boxes
        sorted_positions = np.sort(landed_positions)
        diffs = np.diff(sorted_positions)

        min_spacing = getattr(self._scene_spec, "min_spacing", 0.0)
        box_width = self._scene_spec.box_width

        # Check for too-close or overlapping boxes (only landed ones)
        if np.any(diffs < min_spacing) or np.any(diffs < box_width):
            return True

        return False

    def get_robustness_score(self, state: ConveyorBeltState) -> float:
        """Compute robustness based only on landed boxes."""
        landed_mask = state.falling_heights == 0.0
        landed_positions = state.positions[landed_mask]

        if len(landed_positions) <= 1:
            return float("inf")

        sorted_positions = np.sort(landed_positions)
        diffs = np.diff(sorted_positions)

        min_gap = np.min(diffs)
        min_spacing = getattr(self._scene_spec, "min_spacing", 0.0)
        box_width = self._scene_spec.box_width

        spacing_margin = min_gap - min_spacing
        overlap_margin = min_gap - box_width

        return min(spacing_margin, overlap_margin)
