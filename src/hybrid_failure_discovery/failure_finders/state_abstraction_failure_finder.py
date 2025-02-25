"""A failure finder that uses state abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable, TypeAlias, TypeVar

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
from hybrid_failure_discovery.utils import Trajectory, extend_trajectory_until_failure

AbstractState = TypeVar("AbstractState", bound=Hashable)
AbstractFn: TypeAlias = Callable[[Trajectory], AbstractState]
AbstractHeuristic: TypeAlias =  Callable[[list[AbstractState]], float]


@dataclass
class _Node:
    """Represents a unique sequence of abstract states."""

    abstract_state_sequence: list[AbstractState]
    trajectories: list[Trajectory]
    num_expansions: int = 0

    @property
    def depth(self) -> int:
        """Depth of this node in the tree."""
        return len(self.abstract_state_sequence)

    def __hash__(self) -> int:
        return hash((tuple(self.abstract_state_sequence), self.num_expansions))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, _Node):
            return False
        return (
            self.abstract_state_sequence == other.abstract_state_sequence
            and self.num_expansions == other.num_expansions
        )


class StateAbstractionFailureFinder(FailureFinder):
    """A failure finder that uses state abstractions."""

    def __init__(
        self,
        abstract_fn: AbstractFn,
        abstract_heuristic: AbstractHeuristic,
        max_trajectory_length: int = 100,
        max_num_iters: int = 1000,
        seed: int = 0,
        k: float = 1.0,  # Progressive widening scaling factor
        b: float = 0.5,  # Progressive widening exponent
    ) -> None:
        self._abstract_fn = abstract_fn
        self._abstract_heuristic = abstract_heuristic
        self._max_trajectory_length = max_trajectory_length
        self._max_num_iters = max_num_iters
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._k = k
        self._b = b

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
        root = _Node([abstract_state], [trajectory])
        nodes = [root]

        for itr in range(self._max_num_iters):
            node = self._select_node_with_progressive_widening(nodes, itr)
            print(f"Expanding node with abstract states {node.abstract_state_sequence}")

            failure = self._expand_node(node, nodes, env, controller, failure_monitor)
            node.num_expansions += 1

            if failure is not None:
                print(f"Failure found after {itr} iterations")
                return failure

        print("Failure finding failed.")
        return None

    def _select_node_with_progressive_widening(
        self, nodes: list[_Node], itr: int
    ) -> _Node:
        # I'm not totally sure if this makes sense.
        threshold = self._k * itr**self._b
        eligible_nodes = [node for node in nodes if node.num_expansions <= threshold]
        assert len(eligible_nodes) > 0
        min_depth = min(n.depth for n in eligible_nodes)
        min_depth_eligible_nodes = [n for n in eligible_nodes if n.depth == min_depth]
        return min(min_depth_eligible_nodes, key=lambda n: self._abstract_heuristic(n.abstract_state_sequence))

    def _expand_node(
        self,
        node: _Node,
        nodes: list[_Node],
        env: ConstraintBasedEnvModel[ObsType, ActType],
        controller: ConstraintBasedController[ObsType, ActType],
        failure_monitor: FailureMonitor[ObsType, ActType],
    ) -> Trajectory | None:
        traj = node.trajectories[self._rng.choice(len(node.trajectories))]
        current_abstract_state = node.abstract_state_sequence[-1]
        next_traj, next_abstract_state, failure_found = (
            self._sample_extended_trajectory(
                traj, current_abstract_state, env, controller, failure_monitor
            )
        )

        if failure_found:
            return next_traj

        if next_traj is None:
            raise NotImplementedError("Need to handle this case later")

        # Add the next trajectory to an existing or new node.
        next_abstract_state_sequence = node.abstract_state_sequence + [
            next_abstract_state
        ]
        added_to_existing_node = False

        for node in nodes:
            if node.abstract_state_sequence == next_abstract_state_sequence:
                node.trajectories.append(next_traj)
                added_to_existing_node = True

        if not added_to_existing_node:
            print(
                f"Adding new node with abstract states {next_abstract_state_sequence}"
            )
            new_node = _Node(next_abstract_state_sequence, [next_traj])
            nodes.append(new_node)

        return None

    def _sample_extended_trajectory(
        self,
        current_traj: Trajectory,
        current_abstract_state: AbstractState,
        env: ConstraintBasedEnvModel[ObsType, ActType],
        controller: ConstraintBasedController[ObsType, ActType],
        failure_monitor: FailureMonitor[ObsType, ActType],
    ) -> tuple[Trajectory | None, AbstractState, bool]:

        def _termination_fn(traj: Trajectory) -> bool:
            if len(traj[1]) >= self._max_trajectory_length:
                return True
            next_abstract_state = self._abstract_fn(traj)
            return next_abstract_state != current_abstract_state

        next_traj, failure_found = extend_trajectory_until_failure(
            current_traj, env, controller, failure_monitor, _termination_fn, self._rng
        )
        next_abstract_state = self._abstract_fn(next_traj)
        if next_abstract_state == current_abstract_state:
            return None, next_abstract_state, failure_found
        return next_traj, next_abstract_state, failure_found
