"""A failure finder that uses state abstractions."""

from typing import Callable, Hashable, TypeAlias, TypeVar

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
from hybrid_failure_discovery.utils import Trajectory

AbstractState = TypeVar("AbstractState", bound=Hashable)
AbstractFn: TypeAlias = Callable[[Trajectory], AbstractState]


class StateAbstractionFailureFinder(FailureFinder):
    """A failure finder that uses state abstractions."""

    def __init__(
        self,
        abstract_fn: AbstractFn,
        max_trajectory_length: int = 100,
        max_num_iters: int = 100,
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
    ) -> tuple[list[ObsType], list[ActType]] | None:
        print("Failure finding failed.")
        return None
