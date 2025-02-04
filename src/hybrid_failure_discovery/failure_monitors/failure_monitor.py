"""Base class for failure monitors."""

import abc
from typing import Callable, Generic

from gymnasium.core import ActType, ObsType


class FailureMonitor(Generic[ObsType, ActType]):
    """Monitors a trajectory for failures."""

    @abc.abstractmethod
    def reset(self, initial_state: ObsType) -> None:
        """Reset the monitor given a new initial state."""

    @abc.abstractmethod
    def step(self, action: ActType, state: ObsType) -> bool:
        """Return true if failure and advance any internal state."""


class MemorylessStateFailureMonitor(FailureMonitor[ObsType, ActType]):
    """A failure finder that only checks a given state."""

    def __init__(self, state_check: Callable[[ObsType], bool]) -> None:
        self._state_check = state_check

    def reset(self, initial_state: ObsType) -> None:
        pass

    def step(self, action: ActType, state: ObsType) -> bool:
        return self._state_check(state)
