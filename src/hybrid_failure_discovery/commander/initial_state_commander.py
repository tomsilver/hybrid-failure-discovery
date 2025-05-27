"""Base class for a commander."""

import abc
from typing import Generic

from gymnasium.core import ObsType

from hybrid_failure_discovery.structs import CommandType


class InitialStateCommander(Generic[ObsType, CommandType]):
    """Base class for a commander."""

    @abc.abstractmethod
    def reset(self, initial_state: ObsType) -> None:
        """Reset the commander given a new initial state."""

    @abc.abstractmethod
    def get_command(self) -> CommandType:
        """Get a command for the current state."""

    @abc.abstractmethod
    def update(self, next_state: ObsType) -> None:
        """Update any internal state."""
