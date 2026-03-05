"""Shared utilities for failure discovery."""

import abc
from typing import Any

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from gym_failure_discovery.failure_monitor_wrapper import FailureMonitorWrapper
from gym_failure_discovery.failure_monitors.failure_monitor import FailureMonitor


class Policy(abc.ABC):
    """A stateful policy that maps observations to actions."""

    def reset(self) -> None:
        """Reset any internal state at the start of an episode."""

    @abc.abstractmethod
    def act(self, obs: Any) -> Any:
        """Return an action given the current observation."""


def rollout(
    env: gym.Env[ObsType, ActType],
    monitor: FailureMonitor,
    policy: Policy,
    seed: int,
    max_steps: int,
) -> list[tuple[ObsType, ActType]] | None:
    """Roll out a policy and return the trajectory if a failure occurs.

    Returns a list of (observation, action) pairs leading to the
    failure, or None if no failure was detected within *max_steps*.
    """
    wrapped = FailureMonitorWrapper(env, monitor)
    obs, _ = wrapped.reset(seed=seed)
    policy.reset()
    trajectory: list[tuple[ObsType, ActType]] = []
    for _ in range(max_steps):
        action: ActType = policy.act(obs)
        trajectory.append((obs, action))
        obs, reward, terminated, truncated, _ = wrapped.step(action)
        if reward == 1.0:
            return trajectory
        if terminated or truncated:
            break
    return None
