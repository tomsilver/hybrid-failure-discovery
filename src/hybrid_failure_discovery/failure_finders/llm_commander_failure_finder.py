"""A failure finder that synthesizes a commander using an LLM."""

import inspect

from gymnasium.core import ActType, ObsType
from tomsutils.llm import LargeLanguageModel

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

        # Extract source code.
        env_source = inspect.getsource(env.__class__)
        controller_source = inspect.getsource(controller.__class__)
        failure_monitor_source = inspect.getsource(failure_monitor.__class__)
        commander_source = inspect.getsource(Commander)

        # Create the prompt for the LLM.
        prompt = f"""
Given the following source code:

Environment:
{env_source}

Controller:
{controller_source}

Failure Monitor:
{failure_monitor_source}

Commander Definition:
{commander_source}

Please synthesize a Commander that would induce a failure.

Given your llm_response, I should be able to run the following code:
```python
exec(llm_response, globals())
synthesized_commander = eval('SynthesizedCommander()')
```
"""
        # Use the LLM to generate the Commander.
        synthesized_commander_code, _ = self._llm.query(
            prompt, temperature=1.0, seed=self._seed
        )

        # Execute the synthesized code to create the Commander instance
        exec(synthesized_commander_code, globals())  # pylint: disable=exec-used
        synthesized_commander = eval("SynthesizedCommander()")

        return synthesized_commander
