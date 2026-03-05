"""Failure monitor for the blocks environment.

Detects when a carried block collides with (knocks) a non-held block.
Uses AABB overlap between the held block's current position and each
non-held block's previous position, then checks whether the non-held
block was actually displaced.
"""

from typing import Any

import numpy as np

from gym_failure_discovery.envs.blocks_env import BlocksEnv, BlocksEnvState
from gym_failure_discovery.failure_monitors.failure_monitor import FailureMonitor


class BlocksFailureMonitor(FailureMonitor):
    """Collision monitor for the blocks environment.

    Requires a reference to the ``BlocksEnv`` so it can call
    ``get_state()`` to read the full internal state (block poses,
    held block, robot joints).
    """

    def __init__(
        self,
        env: BlocksEnv,
        move_tol: float = 0.05,
        block_half_extents: tuple[float, float, float] = (0.025, 0.025, 0.025),
    ) -> None:
        self._env = env
        self._move_tol = move_tol
        self._block_half_extents = np.array(block_half_extents)
        self._prev_state: BlocksEnvState | None = None

    def reset(self, obs: Any) -> None:
        self._prev_state = self._env.get_state()

    def step(self, obs: Any, action: Any, next_obs: Any) -> bool:
        assert self._prev_state is not None
        state = self._env.get_state()

        if state.held_block_idx >= 0:
            held_name = self._env.block_names[state.held_block_idx]
            held_pos = np.array(state.get_block_state(held_name).pose.position)
            h2 = 2.0 * self._block_half_extents
            for prev_block in self._prev_state.blocks:
                if prev_block.name == held_name:
                    continue
                prev_pos = np.array(prev_block.pose.position)
                if not np.all(np.abs(held_pos - prev_pos) < h2):
                    continue
                curr_block = state.get_block_state(prev_block.name)
                curr_pos = np.array(curr_block.pose.position)
                if not np.allclose(curr_pos, prev_pos, atol=self._move_tol):
                    return True

        self._prev_state = state
        return False
