"""Base class for failure finders."""

import abc

import gymnasium as gym
import numpy as np

from gym_failure_discovery.failure_monitors.failure_monitor import FailureMonitor


class FailureFinder(abc.ABC):
    """Finds episodes where the failure monitor detects a failure."""

    @abc.abstractmethod
    def find_failure(
        self,
        env: gym.Env,  # type: ignore[type-arg]
        monitor: FailureMonitor,
    ) -> list[tuple[np.ndarray, int]] | None:
        """Search for a failure trajectory.

        Returns a list of (observation, action) pairs if a failure is
        found, or None if the search is exhausted without finding one.
        """
