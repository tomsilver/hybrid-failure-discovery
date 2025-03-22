"""A naive failure finder that just randomly samples trajectories."""

from gymnasium.core import ActType, ObsType
from tomsutils.utils import sample_seed_from_rng

from hybrid_failure_discovery.commander.commander import Commander
from hybrid_failure_discovery.commander.random_commander import RandomCommander
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


class RandomShootingFailureFinder(CommanderFailureFinder):
    """A naive failure finder that just randomly samples trajectories."""

    def get_commander(
        self,
        env: ConstraintBasedEnvModel[ObsType, ActType],
        controller: ConstraintBasedController[ObsType, ActType, CommandType],
        failure_monitor: FailureMonitor[ObsType, ActType, CommandType],
    ) -> Commander[ObsType, ActType, CommandType]:
        seed = sample_seed_from_rng(self._rng)
        command_space = controller.get_command_space()
        commander: RandomCommander[ObsType, ActType, CommandType] = RandomCommander(
            command_space
        )
        commander.seed(seed)
        return commander
