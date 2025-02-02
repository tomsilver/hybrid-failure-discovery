"""A controller for the blocks environment."""

import control as ct
import numpy as np
from gymnasium.spaces import Space
from tomsutils.spaces import FunctionalSpace

from hybrid_failure_discovery.controllers.controller import ConstraintBasedController
from hybrid_failure_discovery.envs.blocks_env import BlocksAction, BlocksEnvState


class BlocksController(
    ConstraintBasedController[BlocksEnvState, BlocksAction]
):
    """A blocks env controller that randomly picks and places blocks."""

    def __init__(
        self,
        seed: int,
    ) -> None:
        super().__init__(seed)

    def reset(self, initial_state: BlocksEnvState) -> None:
        pass

    def step_action_space(self, state: BlocksEnvState) -> Space[BlocksAction]:
        return FunctionalSpace(
            contains_fn=lambda x: isinstance(x, BlocksAction),
            sample_fn=self._sample_action(state)
        )
    
    def _sample_action(self, state: BlocksEnvState) -> BlocksAction:
        import ipdb; ipdb.set_trace()
