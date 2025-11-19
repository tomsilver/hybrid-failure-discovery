# """Tests for conveyorbelt_failure_monitor.py."""

# from hybrid_failure_discovery.envs.conveyorbelt_env import (
#     ConveyorBeltAction,
#     ConveyorBeltEnv,
# )
# from hybrid_failure_discovery.failure_monitors.conveyorbelt_failure_monitor import (
#     ConveyorBeltFailureMonitor,
# )


# def test_conveyorbelt_failure_monitor():
#     """Tests for conveyorbelt_failure_monitor.py."""
#     env = ConveyorBeltEnv(seed=123)
#     monitor = ConveyorBeltFailureMonitor(env.scene_spec)

#     # --- Initial state ---
#     state, _ = env.reset(seed=123)
#     initial_score = monitor.get_robustness_score(state)
#     print("Initial robustness score:", initial_score)

#     # --- Case 1: Overlap (two consecutive drops) ---
#     state, _, _, _, _ = env.step(ConveyorBeltAction(drop_package=True))  # t = 0
#     state, _, _, _, _ = env.step(ConveyorBeltAction(drop_package=True))  # t = 1
#     overlap_score = monitor.get_robustness_score(state)
#     failed_overlap = getattr(monitor, "_check_failures")(state)
#     print("After consecutive drops:")
#     print("  Failure detected:", failed_overlap)
#     print("  Robustness score:", overlap_score)

#     # --- Case 2: Spaced drops (safe) ---
#     env = ConveyorBeltEnv(seed=42)
#     monitor = ConveyorBeltFailureMonitor(env.scene_spec)
#     state, _ = env.reset(seed=42)
#     state, _, _, _, _ = env.step(ConveyorBeltAction(drop_package=True))  # t = 0
#     for _ in range(60):  # let it move forward for ~0.6 s
#         state, _, _, _, _ = env.step(ConveyorBeltAction(drop_package=False))
#     state, _, _, _, _ = env.step(ConveyorBeltAction(drop_package=True))  # second drop
#     spaced_score = monitor.get_robustness_score(state)
#     failed_spaced = getattr(monitor, "_check_failures")(state)
#     print("After spaced drops:")
#     print("  Failure detected:", failed_spaced)
#     print("  Robustness score:", spaced_score)

# def test_first_drop_no_failure_debug():
#     """Smoke check: first drop should NOT trigger failure (only 1 box)."""
#     from hybrid_failure_discovery.envs.conveyorbelt_env import (
#         ConveyorBeltEnv,
#         ConveyorBeltAction,
#     )
#     from hybrid_failure_discovery.failure_monitors.conveyorbelt_failure_monitor import (
#         ConveyorBeltFailureMonitor,
#     )
#     import numpy as np

#     env = ConveyorBeltEnv(seed=0)
#     monitor = ConveyorBeltFailureMonitor(env.scene_spec)
#     state, _ = env.reset(seed=0)

#     # One drop, then no-ops for a few steps (box still falling)
#     state, _, _, _, _ = env.step(ConveyorBeltAction(drop_package=True))
#     for _ in range(3):
#         state, _, _, _, _ = env.step(ConveyorBeltAction(drop_package=False))

#     # Inspect what the monitor would check
#     n_boxes = len(state.positions)
#     positions_sorted = np.sort(state.positions)
#     diffs = np.diff(positions_sorted)
#     min_spacing = getattr(env.scene_spec, "min_spacing", 0.0)
#     box_width = env.scene_spec.box_width

#     # Use the same check the monitor uses
#     failure = False
#     if n_boxes > 1:
#         failure = (np.any(diffs < min_spacing) or np.any(diffs < box_width))

#     print("--- debug ---")
#     print("n_boxes:", n_boxes)
#     print("positions:", state.positions)
#     print("falling_heights:", state.falling_heights)
#     print("sorted_positions:", positions_sorted)
#     print("diffs:", diffs)
#     print("min_spacing:", min_spacing, "box_width:", box_width)
#     print("monitor_failure:", failure)
#     print("robustness:", monitor.get_robustness_score(state))

#     # This SHOULD be False with our monitor (only 1 box)
#     assert n_boxes == 1
#     assert failure is False
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
        """Return True if any pair of landed boxes violates spacing or overlap."""
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
