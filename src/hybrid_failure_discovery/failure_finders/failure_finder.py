"""Base class for failure finders."""

import abc

from gymnasium.core import ActType, ObsType

from hybrid_failure_discovery.commander.commander import Commander
from hybrid_failure_discovery.controllers.controller import ConstraintBasedController
from hybrid_failure_discovery.envs.constraint_based_env_model import (
    ConstraintBasedEnvModel,
)
from hybrid_failure_discovery.failure_monitors.failure_monitor import FailureMonitor
from hybrid_failure_discovery.structs import CommandType
from hybrid_failure_discovery.utils import Trajectory


class FailureFinder(abc.ABC):
    """Base class for failure finders."""

    @abc.abstractmethod
    def run(
        self,
        env: ConstraintBasedEnvModel[ObsType, ActType],
        commander: Commander[ObsType, ActType, CommandType],
        controller: ConstraintBasedController[ObsType, ActType, CommandType],
        failure_monitor: FailureMonitor[ObsType, ActType, CommandType],
    ) -> Trajectory[ObsType, ActType, CommandType] | None:
        """Find a failure trajectory or return None."""
