"""A failure finder that rolls out a given policy."""

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from gym_failure_discovery.failure_finders.failure_finder import FailureFinder
from gym_failure_discovery.failure_monitors.failure_monitor import FailureMonitor
from gym_failure_discovery.utils import Policy, rollout


class OracleFailureFinder(FailureFinder):
    """Rolls out a given policy to find a failure."""

    def __init__(
        self,
        policy: Policy,
        seed: int = 0,
        max_trajectory_length: int = 500,
    ) -> None:
        self._policy = policy
        self._seed = seed
        self._max_trajectory_length = max_trajectory_length

    def find_failure(
        self,
        env: gym.Env[ObsType, ActType],
        monitor: FailureMonitor,
    ) -> list[tuple[ObsType, ActType]] | None:
        return rollout(
            env, monitor, self._policy, self._seed, self._max_trajectory_length
        )
