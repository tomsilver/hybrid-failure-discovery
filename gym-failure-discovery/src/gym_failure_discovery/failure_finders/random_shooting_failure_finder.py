"""A failure finder that randomly samples action sequences."""

import gymnasium as gym
import numpy as np

from gym_failure_discovery.failure_finders.failure_finder import FailureFinder
from gym_failure_discovery.failure_monitor_wrapper import FailureMonitorWrapper
from gym_failure_discovery.failure_monitors.failure_monitor import FailureMonitor


class RandomShootingFailureFinder(FailureFinder):
    """Rolls out random action sequences until a failure is found."""

    def __init__(
        self,
        seed: int = 0,
        max_num_trajectories: int = 100,
        max_trajectory_length: int = 500,
    ) -> None:
        self._seed = seed
        self._max_num_trajectories = max_num_trajectories
        self._max_trajectory_length = max_trajectory_length

    def find_failure(
        self,
        env: gym.Env,  # type: ignore[type-arg]
        monitor: FailureMonitor,
    ) -> list[tuple[np.ndarray, int]] | None:
        wrapped = FailureMonitorWrapper(env, monitor)
        rng = np.random.default_rng(self._seed)
        for _ in range(self._max_num_trajectories):
            obs, _ = wrapped.reset(seed=int(rng.integers(2**31)))
            trajectory: list[tuple[np.ndarray, int]] = []
            for _ in range(self._max_trajectory_length):
                action = int(wrapped.action_space.sample())
                trajectory.append((obs, action))
                obs, reward, terminated, truncated, _ = wrapped.step(action)
                if reward == 1.0:
                    return trajectory
                if terminated or truncated:
                    break
        return None
