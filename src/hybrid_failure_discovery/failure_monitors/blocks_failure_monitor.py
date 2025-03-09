"""Failure monitor for the blocks environment."""

from hybrid_failure_discovery.envs.blocks_env import (
    BlocksAction,
    BlocksCommand,
    BlocksEnvState,
)
from hybrid_failure_discovery.failure_monitors.failure_monitor import (
    FailureMonitor,
)


class BlocksFailureMonitor(FailureMonitor[BlocksEnvState, BlocksAction, BlocksCommand]):
    """A failure occurs when some block that is not held moves."""

    def __init__(self, move_tol: float = 0.05) -> None:
        self._previous_state: BlocksEnvState | None = None
        self._move_tol = move_tol

    def reset(self, initial_state: BlocksEnvState) -> None:
        self._previous_state = initial_state

    def step(
        self, command: BlocksCommand, action: BlocksAction, state: BlocksEnvState
    ) -> bool:
        assert self._previous_state is not None
        for new_block_state in state.blocks:
            block_name = new_block_state.name
            if state.held_block_name == block_name:
                continue
            old_block_state = self._previous_state.get_block_state(block_name)
            if not new_block_state.pose.allclose(
                old_block_state.pose, atol=self._move_tol
            ):
                return True  # failure!
        self._previous_state = state
        return False
