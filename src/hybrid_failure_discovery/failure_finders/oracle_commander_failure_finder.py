"""A failure finder that uses a given commander to sample trajectories."""

from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Space
from tomsutils.utils import sample_seed_from_rng

from hybrid_failure_discovery.commander.commander import Commander
from hybrid_failure_discovery.commander.random_initial_state_commander import (
    RandomInitialStateCommander,
)
from hybrid_failure_discovery.controllers.controller import ConstraintBasedController
from hybrid_failure_discovery.envs.constraint_based_env_model import (
    ConstraintBasedEnvModel,
)
from hybrid_failure_discovery.failure_finders.commander_failure_finder import (
    CommanderFailureFinder,
)
from hybrid_failure_discovery.failure_finders.failure_finder import (
    FailureMonitor,
)
from hybrid_failure_discovery.structs import CommandType


class OracleCommanderFailureFinder(CommanderFailureFinder):
    """A failure finder that uses a given commander to sample trajectories."""

    def __init__(
        self,
        oracle_commander: Commander[ObsType, ActType, CommandType],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._oracle_commander = oracle_commander

    def get_commander(
        self,
        env: ConstraintBasedEnvModel[ObsType, ActType],
        controller: ConstraintBasedController[ObsType, ActType, CommandType],
        failure_monitor: FailureMonitor[ObsType, ActType, CommandType],
        traj_idx: int,
    ) -> Commander[ObsType, ActType, CommandType]:
        return self._oracle_commander  # type: ignore[return-value]

    def get_initial_state(
        self,
        initial_space: Space[ObsType],
        env: ConstraintBasedEnvModel[ObsType, ActType],
        controller: ConstraintBasedController[ObsType, ActType, CommandType],
        failure_monitor: FailureMonitor[ObsType, ActType, CommandType],
    ) -> RandomInitialStateCommander[ObsType]:
        seed = sample_seed_from_rng(self._rng)
        initializer = RandomInitialStateCommander(initial_space)
        initializer.seed(seed)
        return initializer
