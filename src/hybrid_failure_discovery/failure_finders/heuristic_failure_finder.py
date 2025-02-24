"""A failure finder that uses a heuristic over trajectories."""

import heapq as hq
from typing import Callable, TypeAlias

import numpy as np
from gymnasium.core import ActType, ObsType
from tomsutils.utils import sample_seed_from_rng

from hybrid_failure_discovery.controllers.controller import ConstraintBasedController
from hybrid_failure_discovery.envs.constraint_based_env_model import (
    ConstraintBasedEnvModel,
)
from hybrid_failure_discovery.failure_finders.failure_finder import (
    FailureFinder,
    FailureMonitor,
)

Trajectory: TypeAlias = tuple[list[ObsType], list[ActType]]
TrajectoryHeuristic: TypeAlias = Callable[[Trajectory], float]


class HeuristicFailureFinder(FailureFinder):
    """A failure finder that uses a heuristic over trajectories."""

    def __init__(
        self,
        heuristic: TrajectoryHeuristic,
        num_particles: int = 5,
        num_extension_attempts: int = 1,
        extension_termination_prob: float = 0.01,
        max_trajectory_length: int = 100,
        max_num_iters: int = 100,
        seed: int = 0,
    ) -> None:
        self._heuristic = heuristic
        self._num_particles = num_particles
        self._num_extension_attempts = num_extension_attempts
        self._extension_termination_prob = extension_termination_prob
        self._max_trajectory_length = max_trajectory_length
        self._max_num_iters = max_num_iters
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def run(
        self,
        env: ConstraintBasedEnvModel[ObsType, ActType],
        controller: ConstraintBasedController[ObsType, ActType],
        failure_monitor: FailureMonitor[ObsType, ActType],
    ) -> tuple[list[ObsType], list[ActType]] | None:
        # Initialize the particles (partial trajectories).
        initial_states = env.get_initial_states()
        initial_states.seed(sample_seed_from_rng(self._rng))
        particles: list[Trajectory] = [
            ([initial_states.sample()], []) for _ in range(self._num_particles)
        ]
        # Main loop.
        for itr in range(self._max_num_iters):
            # Sample new candidate particles.
            new_candidates: list[Trajectory] = []
            for _ in range(self._num_particles):
                # Sample an existing particle to extend.
                particle = particles[self._rng.choice(len(particles))]
                # Extend the particle in various ways.
                for _ in range(self._num_extension_attempts):
                    new_candidate, failure_found = self._sample_trajectory_extension(
                        particle, env, controller, failure_monitor
                    )
                    # Check if this trajectory is a failure and return if so.
                    if failure_found:
                        print(f"Found a failure after {itr+1} iterations")
                        return new_candidate
                    new_candidates.append(new_candidate)
            # Subselect from the new and old candidates.
            # NOTE: we may need to rejuvenate here to avoid getting stuck.
            # We probably want to think more carefully about this whole planner.
            pool = particles + new_candidates
            particles = self._subselect_particles(pool, self._num_particles)
        print("Failure finding failed.")
        return None

    def _sample_trajectory_extension(
        self,
        trajectory: Trajectory,
        env: ConstraintBasedEnvModel[ObsType, ActType],
        controller: ConstraintBasedController[ObsType, ActType],
        failure_monitor: FailureMonitor[ObsType, ActType],
    ) -> tuple[Trajectory, bool]:
        states, actions = list(trajectory[0]), list(trajectory[1])
        assert len(states) == len(actions) + 1
        # Reset and fast forward the controller and failure monitor.
        failure_monitor.reset(states[0])
        controller.reset(states[0])
        for t in range(len(actions)):
            # NOTE: this makes a strong assumption that controllers are
            # deterministic!! Check this assumption in a hacky way.
            # TODO this is broken because starting planning in the middle of
            # an option might be different from planning from the beginning.
            recovered_action = controller.step(states[t])
            assert env.actions_are_equal(recovered_action, actions[t])
            failure_found = failure_monitor.step(actions[t], states[t + 1])
            assert not failure_found, "Should have already returned"
        # Start the extension.
        state = states[-1]
        while len(actions) < self._max_trajectory_length:
            # Sample an action.
            action_space = controller.step_action_space(state)
            action_space.seed(sample_seed_from_rng(self._rng))
            action = action_space.sample()
            # Update the state.
            next_states = env.get_next_states(state, action)
            next_states.seed(sample_seed_from_rng(self._rng))
            state = next_states.sample()
            # Extend the trajectory.
            actions.append(action)
            states.append(state)
            # Check for failure.
            if failure_monitor.step(action, state):
                return (states, actions), True
            # Check for termination.
            if self._rng.uniform() < self._extension_termination_prob:
                break
        return (states, actions), False

    def _subselect_particles(
        self, pool: list[Trajectory], num_to_select: int
    ) -> list[Trajectory]:
        # NOTE: we should probably compute these and store them along with the
        # trajectories. I'm not sure about tiebreaking though.
        priorities = [(self._heuristic(traj), idx) for idx, traj in enumerate(pool)]
        # We should randomize this later with some temperature.
        selection = hq.nsmallest(num_to_select, priorities)
        return [pool[idx] for _, idx in selection]
