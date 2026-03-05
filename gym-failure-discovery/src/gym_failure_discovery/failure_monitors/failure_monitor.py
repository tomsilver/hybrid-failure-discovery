"""Base class for failure monitors."""

import abc

import numpy as np


class FailureMonitor(abc.ABC):
    """Monitors environment transitions for failures."""

    @abc.abstractmethod
    def reset(self, obs: np.ndarray) -> None:
        """Reset the monitor at the start of an episode."""

    @abc.abstractmethod
    def step(self, obs: np.ndarray, action: int, next_obs: np.ndarray) -> bool:
        """Return True if a failure occurred on this transition."""
