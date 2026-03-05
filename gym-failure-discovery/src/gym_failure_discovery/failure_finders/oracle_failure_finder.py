"""A failure finder that rolls out a given policy."""

from typing import Callable

import gymnasium as gym
import numpy as np

from gym_failure_discovery.failure_finders.failure_finder import FailureFinder
from gym_failure_discovery.failure_monitor_wrapper import FailureMonitorWrapper
from gym_failure_discovery.failure_monitors.failure_monitor import FailureMonitor


class OracleFailureFinder(FailureFinder):
    """Rolls out a given policy to find a failure."""

    def __init__(
        self,
        policy: Callable[[np.ndarray], int],
        seed: int = 0,
        max_trajectory_length: int = 500,
    ) -> None:
        self._policy = policy
        self._seed = seed
        self._max_trajectory_length = max_trajectory_length

    def find_failure(
        self,
        env: gym.Env,  # type: ignore[type-arg]
        monitor: FailureMonitor,
    ) -> list[tuple[np.ndarray, int]] | None:
        wrapped = FailureMonitorWrapper(env, monitor)
        obs, _ = wrapped.reset(seed=self._seed)
        trajectory: list[tuple[np.ndarray, int]] = []
        for _ in range(self._max_trajectory_length):
            action = self._policy(obs)
            trajectory.append((obs, action))
            obs, reward, terminated, truncated, _ = wrapped.step(action)
            if reward == 1.0:
                return trajectory
            if terminated or truncated:
                break
        return None
