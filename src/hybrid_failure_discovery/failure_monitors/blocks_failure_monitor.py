"""Failure monitor for the blocks environment."""

import numpy as np

from hybrid_failure_discovery.controllers.blocks_controller import BlocksCommand
from hybrid_failure_discovery.envs.blocks_env import (
    BlocksAction,
    BlocksEnvState,
)
from hybrid_failure_discovery.failure_monitors.failure_monitor import (
    FailureMonitor,
)


class BlocksFailureMonitor(FailureMonitor[BlocksEnvState, BlocksAction, BlocksCommand]):
    """Failure occurs when the carried block's bounding box overlaps a non-held
    block (geometric intersection), or the robot gets stuck.

    Collision detection uses direct AABB overlap rather than relying on physics
    forces, which are too small to be reliable with teleportation-style control.
    The robot gets "stuck" when its joint positions do not change for
    ``stuck_steps`` consecutive steps, indicating a motion plan was exhausted
    before the current operator's effects were achieved.
    """

    def __init__(
        self,
        move_tol: float = 0.05,
        stuck_steps: int = 5,
        block_half_extents: tuple[float, float, float] = (0.025, 0.025, 0.025),
    ) -> None:
        self._previous_state: BlocksEnvState | None = None
        self._move_tol = move_tol
        self._stuck_steps = stuck_steps
        self._block_half_extents = np.array(block_half_extents)
        self._stuck_count: int = 0
        self._failure_reason: str | None = None

    @property
    def failure_reason(self) -> str | None:
        """Human-readable description of the most recent failure, or None."""
        return self._failure_reason

    @property
    def is_stuck(self) -> bool:
        """True when the robot has been frozen for >= stuck_steps steps.

        Used by extend_trajectory_until_failure to abort a stuck
        trajectory early without counting it as a failure.
        """
        return self._stuck_count >= self._stuck_steps

    def reset_stuck(self) -> None:
        """Reset the stuck counter so a new command gets a fair chance."""
        self._stuck_count = 0

    def reset(self, initial_state: BlocksEnvState) -> None:
        self._previous_state = initial_state
        self._stuck_count = 0
        self._failure_reason = None

    def step(
        self, command: BlocksCommand, action: BlocksAction, state: BlocksEnvState
    ) -> bool:
        assert self._previous_state is not None

        # Collision detection: two-part check.
        #
        # Part 1 — geometric: did the held block enter a non-held block's space?
        #   Compare held block's *current* position against each non-held
        #   block's *previous* position (AABB overlap).  We use *previous*
        #   positions because the env step teleports the held block to the new
        #   position first, then runs 30 physics steps that shove non-held
        #   blocks away, then reads state — so checking current vs current
        #   would always show no overlap.
        #
        # Part 2 — physical: did that block actually get displaced?
        #   If the non-held block did NOT move, the arm just passed over/through
        #   it without consequence (e.g. the robot was approaching its intended
        #   stacking target and placed the block on top successfully).
        #   Only report a failure when the block was knocked out of place.
        if state.held_block_name is not None:
            held_state = state.get_block_state(state.held_block_name)
            held_pos = np.array(held_state.pose.position)
            h2 = 2.0 * self._block_half_extents  # combined AABB half-extents
            for prev_other in self._previous_state.blocks:
                if prev_other.name == state.held_block_name:
                    continue
                prev_pos = np.array(prev_other.pose.position)
                # Part 1: geometric intersection with previous position.
                if not np.all(np.abs(held_pos - prev_pos) < h2):
                    continue
                # Part 2: was this block actually displaced?
                curr_other = state.get_block_state(prev_other.name)
                curr_pos = np.array(curr_other.pose.position)
                if not np.allclose(curr_pos, prev_pos, atol=self._move_tol):
                    self._failure_reason = (
                        f"carried block '{state.held_block_name}' "
                        f"knocked block '{prev_other.name}' out of place"
                    )
                    return True

        # Track whether the robot has been frozen in place for several
        # consecutive steps (motion plan exhausted before effects achieved).
        # This is NOT treated as a failure — the trajectory is allowed to
        # finish naturally so the search can keep looking for real collisions.
        prev_joints = np.array(self._previous_state.robot.joint_positions[:7])
        curr_joints = np.array(state.robot.joint_positions[:7])
        if np.allclose(prev_joints, curr_joints, atol=1e-6):
            self._stuck_count += 1
        else:
            self._stuck_count = 0

        self._previous_state = state
        return False

    def get_robustness_score(self, state: BlocksEnvState) -> float:
        raise NotImplementedError
