"""Base class for a controller."""

import abc
from typing import Generic

from gymnasium.core import ActType, ObsType


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
