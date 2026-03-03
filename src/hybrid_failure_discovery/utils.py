"""Utility functions."""

from typing import Callable

import numpy as np
from gymnasium.core import ActType, ObsType
from tomsutils.utils import sample_seed_from_rng

from hybrid_failure_discovery.commander.commander import Commander
from hybrid_failure_discovery.commander.random_commander import RandomCommander
from hybrid_failure_discovery.controllers.controller import Controller
from hybrid_failure_discovery.envs.constraint_based_env_model import (
    ConstraintBasedEnvModel,
)
from hybrid_failure_discovery.failure_monitors.failure_monitor import FailureMonitor
from hybrid_failure_discovery.structs import CommandType, Trajectory

try:
    from task_then_motion_planning.planning import TaskThenMotionPlanningFailure
except ImportError:

    class TaskThenMotionPlanningFailure(Exception):  # type: ignore[misc, no-redef]
        """Fallback when task_then_motion_planning is not installed."""


def extend_trajectory_until_failure(
    trajectory: Trajectory,
    env: ConstraintBasedEnvModel,
    commander: Commander[ObsType, ActType, CommandType],
    controller: Controller[ObsType, ActType, CommandType],
    failure_monitor: FailureMonitor[ObsType, ActType, CommandType],
    termination_fn: Callable[[Trajectory], bool],
    rng: np.random.Generator,
    catch_task_planning_failure: bool = False,
) -> tuple[Trajectory, bool]:
    """Sample a trajectory extension until failure or termination.

    Returns True if a failure was found.

    If catch_task_planning_failure is True,
    TaskThenMotionPlanningFailure is caught internally and the partial
    trajectory built so far is returned with failure_found=False, rather
    than propagating the exception.  Use this in commander-based finders
    so that last_trajectory is always populated even when the task
    completes successfully before the length limit is reached.
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
        # Skip this check for RandomCommander since it's non-deterministic
        if not isinstance(commander, RandomCommander):
            try:
                recovered_command = commander.get_command()
            except TaskThenMotionPlanningFailure:
                if catch_task_planning_failure:
                    return Trajectory(states, actions, commands), False
                raise
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
        try:
            command = commander.get_command()
        except TaskThenMotionPlanningFailure:
            if catch_task_planning_failure:
                return Trajectory(states, actions, commands), False
            raise
        # Sample an action.
        try:
            action = controller.step(state, command)
        except TaskThenMotionPlanningFailure:
            if catch_task_planning_failure:
                return Trajectory(states, actions, commands), False
            raise
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
        # Allow monitors to signal early termination without a failure (e.g.
        # robot stuck with motion plan exhausted).
        if getattr(failure_monitor, "is_stuck", False):
            return Trajectory(states, actions, commands), False
    return Trajectory(states, actions, commands), False
