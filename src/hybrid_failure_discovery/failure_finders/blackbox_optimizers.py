"""Black-box optimizers to find failures."""

from pdb import set_trace as st

import numpy as np
from gymnasium.core import ActType, ObsType
from tomsutils.utils import sample_seed_from_rng
from tomsutils.spaces import EnumSpace
from hybrid_failure_discovery.controllers.controller import Controller
from hybrid_failure_discovery.envs.constraint_based_env_model import (
    ConstraintBasedEnvModel,
)
from hybrid_failure_discovery.failure_finders.failure_finder import (
    FailureFinder,
    FailureMonitor,
)
from simanneal import Annealer
import numpy as np

# This is the annealing failure finder specific to the hovercraft example
class AnnealingFailureFinder(Annealer, FailureFinder):
    """A naive failure finder that just randomly samples trajectories."""

    def __init__(
        self,
        env, 
        controller, 
        failure_monitor,
        max_num_trajectories: int = 1000,
        max_trajectory_length: int = 100,
        seed: int = 0,
    ) -> None:
        self._max_num_trajectories = max_num_trajectories
        self._max_trajectory_length = max_trajectory_length
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.gx_gy_options = [(-0.42, 0.0), (0.42, 0.0), (0.0, -0.42), (0.0, 0.42)]
        state = self.random_run(env, controller) # input signal; named state in annealer
        Annealer.__init__(self, state)
        self.env=env
        self.failure_monitor=failure_monitor
        self.controller=controller
        self.state=state
        print("Original state:", self.state)

    def move(self):
        """Modify the trajectory slightly."""
        idx = np.random.randint(0, self._max_trajectory_length)  # Choose a random time index to modify
        
        # Sampling random points in the trajectory by replaying it:
        initial_states = self.env.get_initial_states()
        initial_states.seed(sample_seed_from_rng(self._rng))
        hc_state = initial_states.sample()
        self.controller.reset(hc_state)

        # Record the trajectory.
        hc_states = [hc_state]
        actions: list[ActType] = []

        for k in range(self._max_trajectory_length):
            action = self.controller.step(hc_state)
            hc_next_states = self.env.get_next_states(hc_state, action)
            
            if k < idx:
                cmd_gx, cmd_gy = self.gx_gy_options[self.state[k]]
                next_state_matched = False # Flag that needs to turn true as soon as hc_state is set in each iteration with the given goal commands
                for new_hc_state in hc_next_states.elements:
                    if cmd_gx == new_hc_state.gx and cmd_gy == new_hc_state.gy:
                        hc_state = new_hc_state
                        next_state_matched = True
                        break
                if not next_state_matched:
                    # st()
                    idx = k # The new index is not randomly selected, but reset to k. I'm not sure if this is OK with simulated annealing?
            if k >= idx:
                action = self.controller.step(hc_state)
                hc_next_states = self.env.get_next_states(hc_state, action)
                hc_state = hc_next_states.sample()
                cmd_gx, cmd_gy = hc_state.gx, hc_state.gy
                cmd_goal_idx = self.gx_gy_options.index((cmd_gx, cmd_gy))
                self.state[idx] = cmd_goal_idx  # Update one step in the trajectory
            
            # Save the trajectory.
            actions.append(action)
            hc_states.append(hc_state)
        self.actions_traj = actions
        self.hc_state_traj = hc_states
        print("Input Goal Commands: ", self.state)
        return self.energy()  # Return the new energy
    
    def energy(self):
        """Evaluate the cost using `run()`."""        
        # CHeck trajectory for the closest distance to obstacles:
        traj_cost = [self.failure_monitor.get_closest_distance(hc_state) for hc_state in self.hc_state_traj]
        cost = min(traj_cost)
        print("Cost: ", cost)
        return cost

    def select_state(self,
                     next_states,
                     input):
        gx, gy = input
        possible_next_states = []  
        for state in next_states.elements:
            if state.gx == gx and state.gy == gy:
                possible_next_states.append(state)
        state = possible_next_states[0]
        return state
    
    # Initial random run
    def random_run(self, env, controller):
        initial_states = env.get_initial_states()
        initial_states.seed(sample_seed_from_rng(self._rng))
        hc_state = initial_states.sample()
        goal_traj = [self.gx_gy_options.index((hc_state.gx, hc_state.gy))]

        # Reset the controller.
        controller.reset(hc_state)

        hc_states = [hc_state]
        actions: list[ActType] = []
        for _ in range(self._max_trajectory_length):
            # Get the next action.
            action = controller.step(hc_state)
            next_hc_states = env.get_next_states(hc_state, action)
            hc_state = next_hc_states.sample()
            goal_traj.append(self.gx_gy_options.index((hc_state.gx, hc_state.gy)))
            hc_states.append(hc_state)
            actions.append(action)
            
        self.actions_traj = actions
        self.hc_state_traj = hc_states
        return goal_traj

    # Input signal is a dict with elements of the type: {t: {"gx"=-0.42, "gy"=0.0}, ...}
    def run(
        self,
        env: ConstraintBasedEnvModel[ObsType, ActType],
        controller: Controller[ObsType, ActType],
        failure_monitor: FailureMonitor[ObsType, ActType],
        input_signal: dict,
    ) -> tuple[list[ObsType], list[ActType]] | None:
        # for traj_idx in range(self._max_num_trajectories):
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
        t = 0.0
        for _ in range(self._max_trajectory_length):
            # Get the next action.
            action = controller.step(state)
            
            input = input_signal[t] # Input at time t

            t = round(t+0.1,1)

            # Update the state.
            next_states = env.get_next_states(
                state, action
            )  # Possible states of the environment
            
            # next_states.seed(sample_seed_from_rng(self._rng))
            # state = next_states.sample()  # Possible selections by the environment.
            try:
                state = self.select_state(next_states, input) # Select next state that matches the input signal
            except:
                st()

            # Save the trajectory.
            actions.append(action)
            states.append(state)
            # Check for failure.
            if failure_monitor.step(action, state):
                failure_found = True
                break

        # Failure found, we're done!
        if failure_found:
            # print(f"Found a failure after {traj_idx+1} trajectory samples")
            return states, actions
        print("Failure finding failed.")
        
        return None
