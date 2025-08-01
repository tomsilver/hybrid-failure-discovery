"""A failure finder that uses a commander to sample trajectories."""

import abc

import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Space

from hybrid_failure_discovery.commander.commander import Commander
from hybrid_failure_discovery.commander.initial_state_commander import (
    InitialStateCommander,
)
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


class CommanderFailureFinder(FailureFinder):
    """A failure finder that uses a commander to sample trajectories."""

    def __init__(
        self,
        max_num_trajectories: int = 1000,
        max_trajectory_length: int = 100,
        seed: int = 0,
    ) -> None:
        self._max_num_trajectories = max_num_trajectories
        self._max_trajectory_length = max_trajectory_length
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @abc.abstractmethod
    def get_commander(
        self,
        env: ConstraintBasedEnvModel[ObsType, ActType],
        controller: ConstraintBasedController[ObsType, ActType, CommandType],
        failure_monitor: FailureMonitor[ObsType, ActType, CommandType],
        traj_idx: int,
    ) -> Commander[ObsType, ActType, CommandType]:
        """Get a commander for the given environment and controller."""

    @abc.abstractmethod
    def get_initial_state(
        self,
        initial_space: Space[ObsType],
        env: ConstraintBasedEnvModel[ObsType, ActType],
        controller: ConstraintBasedController[ObsType, ActType, CommandType],
        failure_monitor: FailureMonitor[ObsType, ActType, CommandType],
    ) -> InitialStateCommander[ObsType]:
        """Get an initial state commander for the given environment and
        controller."""

    def run(
        self,
        env: ConstraintBasedEnvModel[ObsType, ActType],
        controller: ConstraintBasedController[ObsType, ActType, CommandType],
        failure_monitor: FailureMonitor[ObsType, ActType, CommandType],
    ) -> Trajectory[ObsType, ActType, CommandType] | None:
        for traj_idx in range(self._max_num_trajectories):
            # Initialize the particles (partial trajectories).
            initial_space = env.get_initial_states()
            initializer = self.get_initial_state(
                initial_space, env, controller, failure_monitor
            )
            initial_state = initializer.initialize()

            init_traj: Trajectory[ObsType, ActType, CommandType] = Trajectory(
                [initial_state], [], []
            )

            commander = self.get_commander(env, controller, failure_monitor, traj_idx)

            def _termination_fn(traj: Trajectory) -> bool:
                return len(traj.actions) >= self._max_trajectory_length

            failure_traj, failure_found = extend_trajectory_until_failure(
                init_traj,
                env,
                commander,
                controller,
                failure_monitor,
                _termination_fn,
                self._rng,
            )

            # Failure found, we're done!
            if failure_found:
                print(f"Found a failure after {traj_idx+1} trajectory samples")
                return failure_traj
        print("Failure finding failed.")
        return None
