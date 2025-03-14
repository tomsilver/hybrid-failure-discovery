"""A failure finder that uses a heuristic over trajectories."""

import numpy as np
from gymnasium.core import ActType, ObsType
from scipy.special import logsumexp
from tomsutils.utils import sample_seed_from_rng

from hybrid_failure_discovery.commander.commander import Commander
from hybrid_failure_discovery.commander.random_commander import RandomCommander
from hybrid_failure_discovery.controllers.controller import ConstraintBasedController
from hybrid_failure_discovery.envs.constraint_based_env_model import (
    ConstraintBasedEnvModel,
)
from hybrid_failure_discovery.failure_finders.failure_finder import (
    FailureFinder,
    FailureMonitor,
)
from hybrid_failure_discovery.structs import (
    CommandType,
    Trajectory,
    TrajectoryHeuristic,
)
from hybrid_failure_discovery.utils import extend_trajectory_until_failure


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
        boltzmann_temperature: float = 100.0,
        seed: int = 0,
    ) -> None:
        self._heuristic = heuristic
        self._num_particles = num_particles
        self._num_extension_attempts = num_extension_attempts
        self._extension_termination_prob = extension_termination_prob
        self._max_trajectory_length = max_trajectory_length
        self._max_num_iters = max_num_iters
        self._boltzmann_temperature = boltzmann_temperature
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def run(
        self,
        env: ConstraintBasedEnvModel[ObsType, ActType],
        controller: ConstraintBasedController[ObsType, ActType, CommandType],
        failure_monitor: FailureMonitor[ObsType, ActType, CommandType],
    ) -> Trajectory[ObsType, ActType, CommandType] | None:
        # Initialize the particles (partial trajectories).
        initial_states = env.get_initial_states()
        seed = sample_seed_from_rng(self._rng)
        initial_states.seed(seed)
        particles: list[Trajectory] = [
            Trajectory([initial_states.sample()], [], [])
            for _ in range(self._num_particles)
        ]
        # Get the space of possible commands and create a random commander.
        command_space = controller.get_command_space()
        commander: RandomCommander[ObsType, ActType, CommandType] = RandomCommander(
            command_space
        )
        commander.seed(seed)
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
                        particle, env, commander, controller, failure_monitor
                    )
                    # Check if this trajectory is a failure and return if so.
                    if failure_found:
                        print(f"Found a failure after {itr+1} iterations")
                        return new_candidate
                    # If the new_candidate is of max length, start over.
                    if len(new_candidate.actions) >= self._max_trajectory_length:
                        new_candidate = Trajectory([initial_states.sample()], [], [])
                    new_candidates.append(new_candidate)
            # Subselect from the new and old candidates.
            pool = particles + new_candidates
            particles = self._subselect_particles(pool, self._num_particles)
        print("Failure finding failed.")
        return None

    def _sample_trajectory_extension(
        self,
        trajectory: Trajectory[ObsType, ActType, CommandType],
        env: ConstraintBasedEnvModel[ObsType, ActType],
        commander: Commander[ObsType, ActType, CommandType],
        controller: ConstraintBasedController[ObsType, ActType, CommandType],
        failure_monitor: FailureMonitor[ObsType, ActType, CommandType],
    ) -> tuple[Trajectory[ObsType, ActType, CommandType], bool]:

        def _termination_fn(traj: Trajectory) -> bool:
            if len(traj.actions) >= self._max_trajectory_length:
                return True
            return self._rng.uniform() < self._extension_termination_prob

        return extend_trajectory_until_failure(
            trajectory,
            env,
            commander,
            controller,
            failure_monitor,
            _termination_fn,
            self._rng,
        )

    def _subselect_particles(
        self, pool: list[Trajectory[ObsType, ActType, CommandType]], num_to_select: int
    ) -> list[Trajectory[ObsType, ActType, CommandType]]:
        heuristics = [self._heuristic(traj) for traj in pool]
        log_probs = -self._boltzmann_temperature * np.array(heuristics)
        norm_log_probs = log_probs - logsumexp(log_probs)
        probs = np.exp(norm_log_probs)
        idxs = list(range(len(pool)))
        choices = self._rng.choice(idxs, size=num_to_select, replace=False, p=probs)
        return [pool[idx] for idx in choices]
