"""A failure finder that synthesizes a commander using an LLM."""

import inspect
import re

from gymnasium.core import ActType, ObsType
from tomsutils.llm import LargeLanguageModel, parse_python_code_from_llm_response

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

        # Extract the source code for prompting.
        env_source = inspect.getsource(env.__class__)
        controller_source = inspect.getsource(controller.__class__)
        failure_monitor_source = inspect.getsource(failure_monitor.__class__)
        commander_source = inspect.getsource(Commander)

        # Function to extract import statements from source code.
        def _extract_imports(source_code: str) -> str:
            import_statements = re.findall(r'^\s*(import .+|from .+ import .+)', source_code, re.MULTILINE)
            return '\n'.join(import_statements)

        # Extract import statements from the source code.
        env_imports = _extract_imports(env_source)
        controller_imports = _extract_imports(controller_source)
        failure_monitor_imports = _extract_imports(failure_monitor_source)
        commander_imports = _extract_imports(commander_source)

        # Combine all import statements.
        combined_imports = '\n'.join(set(env_imports.split('\n') + controller_imports.split('\n') + failure_monitor_imports.split('\n') + commander_imports.split('\n')))

        # Determine the module names of the env, controller, and failure_monitor.
        env_module = env.__class__.__module__
        controller_module = controller.__class__.__module__
        failure_monitor_module = failure_monitor.__class__.__module__

        # Generate import statements to import everything from those modules.
        additional_imports = f"from {env_module} import *\nfrom {controller_module} import *\nfrom {failure_monitor_module} import *"
        combined_imports = f"{combined_imports}\n{additional_imports}"

        # Create the prompt for the LLM
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
        response, _ = self._llm.query(
            prompt, temperature=1.0, seed=self._seed
        )
        synthesized_commander_code = parse_python_code_from_llm_response(response)

        # Add imports.
        synthesized_commander_code = f"{combined_imports}\n\n{synthesized_commander_code}"

        # Execute the synthesized code to create the Commander instance.
        exec(synthesized_commander_code, globals())  # pylint: disable=exec-used
        synthesized_commander = eval("SynthesizedCommander()")

        return synthesized_commander
