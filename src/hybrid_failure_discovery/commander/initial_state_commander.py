"""Base class for a commander."""

import abc
from typing import Generic

from gymnasium.core import ObsType

class InitialStateCommander(Generic[ObsType]):
    """Base class for setting the initial state."""

    @abc.abstractmethod
    def initialize(self) -> ObsType:
        """Return the initial state to start the trajectory from."""

