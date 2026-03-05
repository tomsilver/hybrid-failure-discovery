"""Failure monitor for the conveyor belt environment."""

from typing import Any

from gym_failure_discovery.failure_monitors.failure_monitor import FailureMonitor


class ConveyorBeltFailureMonitor(FailureMonitor):
    """Failure occurs when the conveyor belt has exploded."""

    def reset(self, obs: Any) -> None:
        pass

    def step(self, obs: Any, action: Any, next_obs: Any) -> bool:
        return next_obs["exploded"]
