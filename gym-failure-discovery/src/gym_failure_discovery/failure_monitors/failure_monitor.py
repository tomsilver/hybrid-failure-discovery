"""Base class for failure monitors."""

import abc

from gymnasium.core import ActType, ObsType


class FailureMonitor(abc.ABC):
    """Monitors environment transitions for failures."""

    @abc.abstractmethod
    def reset(self, obs: ObsType) -> None:
        """Reset the monitor at the start of an episode."""

    @abc.abstractmethod
    def step(self, obs: ObsType, action: ActType, next_obs: ObsType) -> bool:
        """Return True if a failure occurred on this transition."""
