"""Base class for a controller."""

import abc
from typing import Generic

import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Space
from tomsutils.utils import sample_seed_from_rng


class Controller(Generic[ObsType, ActType]):
    """Base class for a controller."""

    @abc.abstractmethod
    def reset(self, initial_state: ObsType) -> None:
        """Reset the controller given a new initial state."""

    @abc.abstractmethod
    def step(self, state: ObsType) -> ActType:
        """Get the next action to execute and advance internal state.

        NOTE: step() should first be called with the same initial_state given
        to reset().
        """


class ConstraintBasedController(Controller[ObsType, ActType]):
    """A controller that generates a space of possible actions and then samples
    from that space to choose one."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        self._seed = seed
        self._np_random = np.random.default_rng(seed)

    @abc.abstractmethod
    def step_action_space(self, state: ObsType) -> Space[ActType]:
        """Advance any internal state and produce an action space."""

    def step(self, state: ObsType) -> ActType:
        action_space = self.step_action_space(state)
        action_space.seed(sample_seed_from_rng(self._np_random))
        action = action_space.sample()
        return action
