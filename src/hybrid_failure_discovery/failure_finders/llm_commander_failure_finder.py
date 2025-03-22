"""A failure finder that synthesizes a commander using an LLM."""

from gymnasium.core import ActType, ObsType

from hybrid_failure_discovery.commander.commander import Commander
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
from tomsutils.llm import LargeLanguageModel


class LLMCommanderFailureFinder(CommanderFailureFinder):
    """A failure finder that synthesizes a commander using an LLM."""

    def __init__(
        self,
        llm: LargeLanguageModel,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._llm = llm

    def get_commander(
        self,
        env: ConstraintBasedEnvModel[ObsType, ActType],
        controller: ConstraintBasedController[ObsType, ActType, CommandType],
        failure_monitor: FailureMonitor[ObsType, ActType, CommandType],
    ) -> Commander[ObsType, ActType, CommandType]:
        
        # Build LLM prompt.
        env_description = env.get_description()
        controller_description = controller.get_description()
        failure_monitor = failure_monitor.get_description()

        import ipdb; ipdb.set_trace()
