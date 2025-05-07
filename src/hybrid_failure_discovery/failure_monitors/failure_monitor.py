"""Base class for failure monitors."""

import abc
from typing import Callable, Generic

from gymnasium.core import ActType, ObsType

from hybrid_failure_discovery.structs import CommandType


class FailureMonitor(Generic[ObsType, ActType, CommandType]):
    """Monitors a trajectory for failures."""

    @abc.abstractmethod
    def reset(self, initial_state: ObsType) -> None:
        """Reset the monitor given a new initial state."""

    @abc.abstractmethod
    def step(self, command: CommandType, action: ActType, state: ObsType) -> bool:
        """Return true if failure and advance any internal state."""

    @abc.abstractmethod
    def get_robustness_score(self, state: ObsType) -> float:
        """Lower means closer to failure."""


class MemorylessStateFailureMonitor(
    FailureMonitor[ObsType, ActType, CommandType], abc.ABC
):
    """A failure finder that only checks a given state."""

    def __init__(self, state_check: Callable[[ObsType], bool]) -> None:
        self._state_check = state_check

    def reset(self, initial_state: ObsType) -> None:
        pass

    def step(self, command: CommandType, action: ActType, state: ObsType) -> bool:
        return self._state_check(state)
