"""A failure finder that randomly samples action sequences."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType

from gym_failure_discovery.failure_finders.failure_finder import FailureFinder
from gym_failure_discovery.failure_monitors.failure_monitor import FailureMonitor
from gym_failure_discovery.utils import Policy, rollout


class _RandomPolicy(Policy):
    """Samples actions uniformly from the action space."""

    def __init__(self, action_space: gym.Space[ActType]) -> None:
        self._action_space = action_space

    def act(self, obs: Any) -> Any:
        return self._action_space.sample()


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
        env: gym.Env[ObsType, ActType],
        monitor: FailureMonitor,
    ) -> list[tuple[ObsType, ActType]] | None:
        policy = _RandomPolicy(env.action_space)
        rng = np.random.default_rng(self._seed)
        for _ in range(self._max_num_trajectories):
            env_seed = int(rng.integers(2**31))
            result = rollout(
                env, monitor, policy, env_seed, self._max_trajectory_length
            )
            if result is not None:
                return result
        return None
