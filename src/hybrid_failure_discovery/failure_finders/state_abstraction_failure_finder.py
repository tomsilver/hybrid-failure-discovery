"""A failure finder that uses state abstractions."""
from __future__ import annotations

from typing import Callable, Hashable, TypeAlias, TypeVar

import numpy as np
from gymnasium.core import ActType, ObsType
from tomsutils.utils import sample_seed_from_rng
from dataclasses import dataclass

from hybrid_failure_discovery.controllers.controller import ConstraintBasedController
from hybrid_failure_discovery.envs.constraint_based_env_model import (
    ConstraintBasedEnvModel,
)
from hybrid_failure_discovery.failure_finders.failure_finder import (
    FailureFinder,
    FailureMonitor,
)
from hybrid_failure_discovery.utils import Trajectory

AbstractState = TypeVar("AbstractState", bound=Hashable)
AbstractFn: TypeAlias = Callable[[Trajectory], AbstractState]


@dataclass
class _Node:
    """Represents a unique sequence of abstract states."""

    abstract_state_sequence: list[AbstractState]
    trajectories: list[Trajectory]
    children: list[_Node]



class StateAbstractionFailureFinder(FailureFinder):
    """A failure finder that uses state abstractions."""

    def __init__(
        self,
        abstract_fn: AbstractFn,
        max_trajectory_length: int = 100,
        max_num_iters: int = 10,
        seed: int = 0,
    ) -> None:
        self._abstract_fn = abstract_fn
        self._max_trajectory_length = max_trajectory_length
        self._max_num_iters = max_num_iters
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def run(
        self,
        env: ConstraintBasedEnvModel[ObsType, ActType],
        controller: ConstraintBasedController[ObsType, ActType],
        failure_monitor: FailureMonitor[ObsType, ActType],
    ) -> Trajectory | None:
        # NOTE: sticking with one initial state for now. In the future, should
        # also progressively sample more initial states.
        initial_states = env.get_initial_states()
        initial_states.seed(sample_seed_from_rng(self._rng))
        initial_state = initial_states.sample()
        trajectory = ([initial_state], [])
        abstract_state = self._abstract_fn(trajectory)
        root = _Node([abstract_state], [trajectory], children=[])
        current_nodes = [root]
        # Progressive widening: at every iteration, all nodes should have the
        # same number of trajectories. But note that this is not necessarily the
        # same number of children because of the abstract state grouping.
        # Nonetheless, this is not going to scale well; we should add heuristics.
        for itr in range(self._max_num_iters):
            next_nodes = list(current_nodes)
            for node in current_nodes:
                failure = self._expand_node(node, next_nodes, itr, env, controller, failure_monitor)
                if failure is not None:
                    print(f"Failure found after {itr} iterations")
                    return failure
        print("Failure finding failed.")
        return None

    def _expand_node(self, node: _Node, next_nodes: list[_Node], num_children: int,
                     env: ConstraintBasedEnvModel[ObsType, ActType],
                            controller: ConstraintBasedController[ObsType, ActType],
                            failure_monitor: FailureMonitor[ObsType, ActType],
                    ) -> Trajectory | None:
        import ipdb; ipdb.set_trace()