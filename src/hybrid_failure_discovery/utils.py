"""Utility functions."""

from typing import Callable

import numpy as np
from gymnasium.core import ActType, ObsType
from tomsutils.utils import sample_seed_from_rng

from hybrid_failure_discovery.commander.commander import Commander
from hybrid_failure_discovery.controllers.controller import Controller
from hybrid_failure_discovery.envs.constraint_based_env_model import (
    ConstraintBasedEnvModel,
)
from hybrid_failure_discovery.failure_monitors.failure_monitor import FailureMonitor
from hybrid_failure_discovery.structs import CommandType, Trajectory


def extend_trajectory_until_failure(
    trajectory: Trajectory,
    env: ConstraintBasedEnvModel,
    commander: Commander[ObsType, ActType, CommandType],
    controller: Controller[ObsType, ActType, CommandType],
    failure_monitor: FailureMonitor[ObsType, ActType, CommandType],
    termination_fn: Callable[[Trajectory], bool],
    rng: np.random.Generator,
) -> tuple[Trajectory, bool]:
    """Sample a trajectory extension until failure or termination.

    Returns True if a failure was found.
    """
    states = list(trajectory.observations)
    actions = list(trajectory.actions)
    commands = list(trajectory.commands)
    trajectory = Trajectory(states, actions, commands)
    assert len(states) == len(commands) + 1 == len(actions) + 1
    # Reset and fast forward the controller and failure monitor.
    failure_monitor.reset(states[0])
    controller.reset(states[0])
    commander.reset(states[0])
    for t in range(len(actions)):
        # NOTE: this makes a strong assumption that controllers are
        # deterministic!! Check this assumption in a hacky way.
        recovered_command = commander.get_command()
        assert recovered_command == commands[t]
        recovered_action = controller.step(states[t], commands[t])
        assert env.actions_are_equal(recovered_action, actions[t])
        failure_found = failure_monitor.step(commands[t], actions[t], states[t + 1])
        assert not failure_found, "Should have already returned"
        commander.update(actions[t], states[t + 1])
    # Start the extension.
    state = states[-1]
    while not termination_fn(trajectory):
        # Sample a command.
        command = commander.get_command()
        # Sample an action.
        action = controller.step(state, command)
        # Update the state.
        next_states = env.get_next_states(state, action)
        next_states.seed(sample_seed_from_rng(rng))
        state = next_states.sample()
        # Extend the trajectory.
        commander.update(action, state)
        actions.append(action)
        states.append(state)
        commands.append(command)
        # Check for failure.
        if failure_monitor.step(command, action, state):
            return Trajectory(states, actions, commands), True
    return Trajectory(states, actions, commands), False
