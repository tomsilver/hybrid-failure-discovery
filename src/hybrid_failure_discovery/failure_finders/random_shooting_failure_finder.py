"""A naive failure finder that just randomly samples trajectories."""

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

# Sample 
class RandomShootingFailureFinder(FailureFinder):
    """A naive failure finder that just randomly samples trajectories."""

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

    def run(
        self,
        env: ConstraintBasedEnvModel[ObsType, ActType],
        controller: ConstraintBasedController[ObsType, ActType],
        failure_monitor: FailureMonitor[ObsType, ActType],
    ) -> tuple[list[ObsType], list[ActType]] | None:
        for traj_idx in range(self._max_num_trajectories):
            # Sample an initial state.
            initial_states = env.get_initial_states()
            initial_states.seed(sample_seed_from_rng(self._rng))
            state = initial_states.sample()
            # Reset the controller and monitor.
            controller.reset(state)
            failure_monitor.reset(state)
            # Record the trajectory.
            states = [state]
            actions: list[ActType] = []
            failure_found = False
            for _ in range(self._max_trajectory_length):
                # Sample an action.
                action_space = controller.step_action_space(state)
                action_space.seed(sample_seed_from_rng(self._rng))
                action = action_space.sample()
                # Update the state.
                next_states = env.get_next_states(
                    state, action
                )  # Possible states of the environment
                next_states.seed(sample_seed_from_rng(self._rng))
                state = next_states.sample()  # Possible selections by the environment.
                # Save the trajectory.
                actions.append(action)
                states.append(state)
                # Check for failure.
                if failure_monitor.step(action, state):
                    failure_found = True
                    break
            # Failure found, we're done!
            if failure_found:
                print(f"Found a failure after {traj_idx+1} trajectory samples")
                return states, actions
        print("Failure finding failed.")
        return None
