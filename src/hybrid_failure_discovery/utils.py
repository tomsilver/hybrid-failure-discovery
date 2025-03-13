"""Utility functions."""

from typing import Callable, TypeAlias

import numpy as np
from gymnasium.core import ActType, ObsType
from tomsutils.utils import sample_seed_from_rng

from hybrid_failure_discovery.controllers.controller import Controller
from hybrid_failure_discovery.envs.constraint_based_env_model import (
    ConstraintBasedEnvModel,
)
from hybrid_failure_discovery.failure_monitors.failure_monitor import FailureMonitor

Trajectory: TypeAlias = tuple[list[ObsType], list[ActType]]
TrajectoryHeuristic: TypeAlias = Callable[[Trajectory], float]


def extend_trajectory_until_failure(
    trajectory: Trajectory,
    env: ConstraintBasedEnvModel,
    controller: Controller,
    failure_monitor: FailureMonitor,
    termination_fn: Callable[[Trajectory], bool],
    rng: np.random.Generator,
) -> tuple[Trajectory, bool]:
    """Sample a trajectory extension until failure or termination.

    Returns True if a failure was found.
    """
    states, actions = list(trajectory[0]), list(trajectory[1])
    assert len(states) == len(actions) + 1
    # Reset and fast forward the controller and failure monitor.
    failure_monitor.reset(states[0])
    controller.reset(states[0])
    for t in range(len(actions)):
        # NOTE: this makes a strong assumption that controllers are
        # deterministic!! Check this assumption in a hacky way.
        recovered_action = controller.step(states[t])
        assert env.actions_are_equal(recovered_action, actions[t])
        failure_found = failure_monitor.step(actions[t], states[t + 1])
        assert not failure_found, "Should have already returned"
    # Start the extension.
    state = states[-1]
    while not termination_fn((states, actions)):
        # Sample an action.
        action = controller.step(state)
        # Update the state.
        next_states = env.get_next_states(state, action)
        next_states.seed(sample_seed_from_rng(rng))
        state = next_states.sample()
        # Extend the trajectory.
        actions.append(action)
        states.append(state)
        # Check for failure.
        if failure_monitor.step(action, state):
            return (states, actions), True
    return (states, actions), False
