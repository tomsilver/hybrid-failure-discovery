"""A controller for the blocks environment."""

import abc

import numpy as np
from gymnasium.spaces import Space
from tomsutils.spaces import FunctionalSpace

from hybrid_failure_discovery.controllers.controller import ConstraintBasedController
from hybrid_failure_discovery.envs.blocks_env import (
    BlocksAction,
    BlocksEnv,
    BlocksEnvSceneSpec,
    BlocksEnvState,
)


class BlocksController(ConstraintBasedController[BlocksEnvState, BlocksAction]):
    """A blocks env controller that randomly picks and places blocks."""

    def __init__(
        self,
        seed: int,
        scene_spec: BlocksEnvSceneSpec,
    ) -> None:
        super().__init__(seed)
        self._scene_spec = scene_spec

        # Create a "simulator".
        self._sim = BlocksEnv(scene_spec, seed=seed)

        # Create options.
        self._options = [
            PickBlockOption(seed, f"block{i}", self._sim)
            for i in range(scene_spec.num_blocks)
        ]

        # Track the current option.
        self._current_option: BlocksOption | None = None

    def reset(self, initial_state: BlocksEnvState) -> None:
        self._current_option = None

    def step_action_space(self, state: BlocksEnvState) -> Space[BlocksAction]:
        return FunctionalSpace(
            contains_fn=lambda x: isinstance(x, BlocksAction),
            sample_fn=self._sample_action(state),
        )

    def _sample_action(self, state: BlocksEnvState) -> BlocksAction:
        if self._current_option is None or self._current_option.terminate(state):
            # Sample a new initiable option.
            idxs = list(range(len(self._options)))
            self._np_random.shuffle(idxs)
            ordered_options = [self._options[i] for i in idxs]
            for option in ordered_options:
                if option.can_initiate(state):
                    self._current_option = option
                    self._current_option.initiate(state)
                    break
            assert self._current_option is not None
        return self._current_option.step(state)


class BlocksOption:
    """A partial controller that initiates and then terminates, e.g., pick."""

    @abc.abstractmethod
    def can_initiate(self, state: BlocksEnvState) -> bool:
        """Whether the option could be initiated in the given state."""

    @abc.abstractmethod
    def initiate(self, state: BlocksEnvState) -> bool:
        """Initiate the option."""

    @abc.abstractmethod
    def step(self, state: BlocksEnvState) -> BlocksAction:
        """Advance any internal state and produce an action."""

    @abc.abstractmethod
    def terminate(self, state: BlocksEnvState) -> bool:
        """Whether the option terminates in the given state."""


class PickBlockOption(BlocksOption):
    """A partial controller for picking an exposed block."""

    def __init__(self, seed: int, block_name: str, sim: BlocksEnv) -> None:
        self._seed = seed
        self._block_name = block_name
        self._sim = sim
        self._rng = np.random.default_rng(seed)
        self._plan: list[BlocksAction] = []

    def can_initiate(self, state: BlocksEnvState) -> bool:
        # Robot hand must be empty.
        if state.held_block_name:
            return False

        # The block must not have anything on top of it.
        block_state = state.get_block_state(self._block_name)
        x_thresh = self._sim.scene_spec.block_half_extents[0]
        y_thresh = self._sim.scene_spec.block_half_extents[1]
        for other_block_state in state.blocks:
            if other_block_state.name == self._block_name:
                continue
            # Check if this other block is above the main one.
            if other_block_state.pose.position[2] <= block_state.pose.position[2]:
                continue
            # Check if close enough in the x plane.
            if (
                abs(other_block_state.pose.position[0] - block_state.pose.position[0])
                > x_thresh
            ):
                continue
            # Check if close enough in the y plane.
            if (
                abs(other_block_state.pose.position[1] - block_state.pose.position[1])
                > y_thresh
            ):
                continue
            # On is true.
            return False

        return True

    def initiate(self, state: BlocksEnvState) -> bool:
        # TODO make a plan
        import ipdb

        ipdb.set_trace()

    def step(self, state: BlocksEnvState) -> BlocksAction:
        return self._plan.pop(0)

    def terminate(self, state):
        return not self._plan
