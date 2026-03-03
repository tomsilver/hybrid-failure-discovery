"""Environment-agnostic failure finder guided by the monitor's robustness
score."""

from typing import Callable

import numpy as np
from gymnasium.core import ActType, ObsType
from scipy.special import logsumexp
from tomsutils.utils import sample_seed_from_rng

try:
    from task_then_motion_planning.planning import TaskThenMotionPlanningFailure
except ImportError:

    class TaskThenMotionPlanningFailure(Exception):  # type: ignore[misc, no-redef]
        """Fallback when task_then_motion_planning is not installed."""


from hybrid_failure_discovery.commander.random_commander import RandomCommander
from hybrid_failure_discovery.controllers.controller import ConstraintBasedController
from hybrid_failure_discovery.envs.constraint_based_env_model import (
    ConstraintBasedEnvModel,
)
from hybrid_failure_discovery.failure_finders.failure_finder import (
    FailureFinder,
    FailureMonitor,
)
from hybrid_failure_discovery.structs import CommandType, Trajectory
from hybrid_failure_discovery.utils import extend_trajectory_until_failure


class AdaptiveRobustnessFailureFinder(FailureFinder):
    """Environment-agnostic failure finder guided by robustness scores.

    Maintains a population of candidate trajectories and iteratively extends
    them, biasing selection toward trajectories whose states have the lowest
    robustness score (i.e. closest to failure) as reported by the failure
    monitor.

    When ``get_robustness_score`` raises ``NotImplementedError`` (not yet
    implemented for an environment) the finder falls back to uniform random
    selection, making it equivalent in quality to
    ``RandomShootingFailureFinder`` in that case.

    No environment-specific knowledge is required: the finder works with any
    ``ConstraintBasedEnvModel`` / ``FailureMonitor`` pair.

    Args:
        num_particles: Number of trajectory candidates kept in the population.
        num_extension_attempts: How many extensions to try per particle per
            iteration.
        extension_termination_prob: Per-step probability of stopping an
            extension early (encourages trajectory length diversity).
        max_trajectory_length: Hard cap on trajectory length; a particle that
            reaches this length is reset to a fresh initial state.
        max_num_iters: Maximum number of population update iterations.
        boltzmann_temperature: Controls how strongly robustness scores bias
            selection.  Higher values concentrate selection on the best
            particle; 0 gives uniform selection.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_particles: int = 10,
        num_extension_attempts: int = 3,
        extension_termination_prob: float = 0.01,
        max_trajectory_length: int = 1000,
        max_num_iters: int = 50,
        boltzmann_temperature: float = 1.0,
        seed: int = 0,
    ) -> None:
        self._num_particles = num_particles
        self._num_extension_attempts = num_extension_attempts
        self._extension_termination_prob = extension_termination_prob
        self._max_trajectory_length = max_trajectory_length
        self._max_num_iters = max_num_iters
        self._boltzmann_temperature = boltzmann_temperature
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._last_trajectory: Trajectory | None = None

    @property
    def last_trajectory(self) -> Trajectory | None:
        """The last fully-run trajectory regardless of whether it was a
        failure."""
        return self._last_trajectory

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        env: ConstraintBasedEnvModel[ObsType, ActType],
        controller: ConstraintBasedController[ObsType, ActType, CommandType],
        failure_monitor: FailureMonitor[ObsType, ActType, CommandType],
    ) -> Trajectory[ObsType, ActType, CommandType] | None:
        initial_states = env.get_initial_states()
        init_seed = sample_seed_from_rng(self._rng)
        initial_states.seed(init_seed)

        # Initialise particle population with random starting states.
        particles: list[Trajectory] = [
            Trajectory([initial_states.sample()], [], [])
            for _ in range(self._num_particles)
        ]

        command_space = controller.get_command_space()
        commander: RandomCommander[ObsType, ActType, CommandType] = RandomCommander(
            command_space
        )
        commander.seed(init_seed)

        for itr in range(self._max_num_iters):
            new_candidates: list[Trajectory] = []

            for particle_idx in range(self._num_particles):
                # Boltzmann-weighted selection from current population.
                particle = self._select_particle(particles, failure_monitor)

                for attempt in range(self._num_extension_attempts):
                    try:
                        new_candidate, failure_found = extend_trajectory_until_failure(
                            particle,
                            env,
                            commander,
                            controller,
                            failure_monitor,
                            self._make_termination_fn(),
                            self._rng,
                        )
                    except TaskThenMotionPlanningFailure:
                        # Task completed without failure — restart this particle.
                        new_candidate = Trajectory([initial_states.sample()], [], [])
                        failure_found = False

                    if failure_found:
                        self._last_trajectory = new_candidate
                        total_attempts = (
                            itr * self._num_particles * self._num_extension_attempts
                            + particle_idx * self._num_extension_attempts
                            + attempt
                            + 1
                        )
                        print(
                            f"Found a failure after {itr + 1} iteration(s) "
                            f"(particle {particle_idx + 1}/{self._num_particles}, "
                            f"attempt {attempt + 1}/{self._num_extension_attempts}, "
                            f"total attempts: ~{total_attempts})"
                        )
                        return new_candidate

                    if len(new_candidate.actions) > 0:
                        self._last_trajectory = new_candidate

                    # Reset particles that have exhausted their length budget.
                    if len(new_candidate.actions) >= self._max_trajectory_length:
                        new_candidate = Trajectory([initial_states.sample()], [], [])

                    new_candidates.append(new_candidate)

            # Subselect elite particles from the combined old + new pool.
            pool = particles + new_candidates
            particles = self._subselect_particles(
                pool, self._num_particles, failure_monitor
            )

        print("Failure finding failed.")
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trajectory_score(
        self,
        traj: Trajectory,
        failure_monitor: FailureMonitor,
    ) -> float:
        """Return the minimum robustness score over all states in the
        trajectory.

        Lower score means closer to failure.  Falls back to 0.0 (neutral) if
        ``get_robustness_score`` is not implemented, so all trajectories are
        treated equally and selection remains uniform.
        """
        try:
            scores = [
                failure_monitor.get_robustness_score(s) for s in traj.observations
            ]
            min_score = float(min(scores))
            if not np.isfinite(min_score):
                return 0.0
            return min_score
        except NotImplementedError:
            return 0.0

    def _select_particle(
        self,
        particles: list[Trajectory],
        failure_monitor: FailureMonitor,
    ) -> Trajectory:
        """Sample one particle, weighted by Boltzmann probabilities over
        scores."""
        scores = np.array(
            [self._trajectory_score(p, failure_monitor) for p in particles]
        )
        probs = self._boltzmann_probs(scores)
        idx = int(self._rng.choice(len(particles), p=probs))
        return particles[idx]

    def _subselect_particles(
        self,
        pool: list[Trajectory],
        num_to_select: int,
        failure_monitor: FailureMonitor,
    ) -> list[Trajectory]:
        """Select ``num_to_select`` particles from the pool without
        replacement."""
        scores = np.array([self._trajectory_score(p, failure_monitor) for p in pool])
        probs = self._boltzmann_probs(scores)
        n = min(num_to_select, len(pool))
        idxs = self._rng.choice(len(pool), size=n, replace=False, p=probs)
        return [pool[i] for i in idxs]

    def _boltzmann_probs(self, scores: np.ndarray) -> np.ndarray:
        """Convert an array of scores to Boltzmann selection probabilities.

        Lower score → higher probability.  Falls back to uniform when
        all scores are equal or when numerical issues arise.
        """
        log_probs = -self._boltzmann_temperature * scores
        try:
            norm_log_probs = log_probs - logsumexp(log_probs)
            probs = np.exp(norm_log_probs)
        except (FloatingPointError, ValueError):
            probs = np.ones(len(scores)) / len(scores)

        # Guard against NaN / non-finite values from degenerate score arrays.
        if not np.all(np.isfinite(probs)) or probs.sum() == 0:
            probs = np.ones(len(scores)) / len(scores)

        # Re-normalise to absorb any floating-point drift.
        probs = probs / probs.sum()
        return probs

    def _make_termination_fn(self) -> Callable[[Trajectory], bool]:
        """Return a fresh termination function that uses the shared RNG."""

        def _termination_fn(traj: Trajectory) -> bool:
            if len(traj.actions) >= self._max_trajectory_length:
                return True
            return bool(self._rng.uniform() < self._extension_termination_prob)

        return _termination_fn
