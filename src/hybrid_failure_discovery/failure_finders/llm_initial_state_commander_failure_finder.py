"""A failure finder that synthesizes a commander using an LLM."""

import ast
import inspect
from typing import Any

from gymnasium.core import ActType, ObsType
from tomsutils.llm import LargeLanguageModel, parse_python_code_from_llm_response
from tomsutils.utils import sample_seed_from_rng

from hybrid_failure_discovery.commander.commander import Commander
from hybrid_failure_discovery.commander.initial_state_commander import (
    InitialStateCommander,
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


class LLMCommanderFailureFinder(CommanderFailureFinder):
    """A failure finder that synthesizes a commander using an LLM."""

    def __init__(
        self,
        llm: LargeLanguageModel,
        *args,
        llm_temperature: float = 0.7,
        num_synthesis_retries: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._llm = llm
        self._llm_temperature = llm_temperature
        self._num_synthesis_retries = num_synthesis_retries

    def get_commander(
        self,
        env: ConstraintBasedEnvModel[ObsType, ActType],
        controller: ConstraintBasedController[ObsType, ActType, CommandType],
        failure_monitor: FailureMonitor[ObsType, ActType, CommandType],
        traj_idx: int,
    ) -> Commander[ObsType, ActType, CommandType]:

        previous_attempt_error: str | None = None
        for _ in range(self._num_synthesis_retries):
            llm_seed = sample_seed_from_rng(self._rng)
            commander, previous_attempt_error = self._synthesize_commander_with_llm(
                env, controller, failure_monitor, llm_seed, previous_attempt_error
            )
            if commander is not None:
                return commander
        raise RuntimeError("Failed to synthesize a commander with the LLM.")

    def get_initial_state_commander(
        self,
        env: ConstraintBasedEnvModel[ObsType, ActType],
        controller: ConstraintBasedController[ObsType, ActType, CommandType],
        failure_monitor: FailureMonitor[ObsType, ActType, CommandType],
    ) -> InitialStateCommander[ObsType, CommandType]:
        pass

    def _synthesize_commander_with_llm(
        self,
        env: ConstraintBasedEnvModel[ObsType, ActType],
        controller: ConstraintBasedController[ObsType, ActType, CommandType],
        failure_monitor: FailureMonitor[ObsType, ActType, CommandType],
        llm_seed: int,
        previous_attempt_error: str | None = None,
    ) -> tuple[Commander[ObsType, ActType, CommandType] | None, str]:
        # Extract the source code for prompting.
        def _get_source_code_from_obj(obj: Any) -> str:
            module = inspect.getmodule(obj.__class__)
            assert module is not None
            return inspect.getsource(module)

        env_source = _get_source_code_from_obj(env)
        controller_source = _get_source_code_from_obj(controller)
        failure_monitor_source = _get_source_code_from_obj(failure_monitor)
        commander_source = inspect.getsource(Commander)
        initial_state_commander_source = inspect.getsource(InitialStateCommander)

        # Function to extract import statements from source code.
        def _extract_imports(source_code: str) -> str:
            tree = ast.parse(source_code)
            import_statements = []

            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    statement = ast.unparse(node)
                    if "__future__ import annotations" in statement:
                        continue
                    import_statements.append(statement)

            return "\n".join(import_statements)

        # Extract import statements from the source code.
        env_imports = _extract_imports(env_source)
        controller_imports = _extract_imports(controller_source)
        failure_monitor_imports = _extract_imports(failure_monitor_source)
        commander_imports = _extract_imports(commander_source)
        initial_state_imports = _extract_imports(initial_state_commander_source)

        # Combine all import statements.
        combined_imports = "from __future__ import annotations\n"
        combined_imports += "\n".join(
            set(
                env_imports.split("\n")
                + controller_imports.split("\n")
                + failure_monitor_imports.split("\n")
                + commander_imports.split("\n")
                + initial_state_imports.split("\n")
            )
        )

        # Determine the module names of the env, controller, and failure_monitor.
        env_module = env.__class__.__module__
        controller_module = controller.__class__.__module__
        failure_monitor_module = failure_monitor.__class__.__module__
        commander_module = Commander.__module__
        initial_state_commander_module = InitialStateCommander.__module__

        # Generate import statements to import everything from those modules.
        additional_imports = f"from {env_module} import *\nfrom {controller_module} import *\nfrom {failure_monitor_module} import *\nfrom {commander_module} import Commander \nfrom {initial_state_commander_module} import InitialStateCommander"  # pylint: disable=line-too-long
        combined_imports = f"{combined_imports}\n{additional_imports}"

        # Create the prompt for the LLM.
        prompt = f"""
Given the following source code:

Environment:
{env_source}

Controller:
{controller_source}

Failure Monitor:
{failure_monitor_source}

Commander Definition (NOTE: use `from {commander_module} import Commander`):
{commander_source}

Initial State Commander Definition (NOTE: use `from {initial_state_commander_module} import InitialStateCommander`):
{commander_source}

Please synthesize an InitialStateCommander and a Commander that would induce a failure.

Your classes should be called SynthesizedInitialStateCommander() and SynthesizedCommander(), respectively, and should take no arguments in the constructor.
"""
        if previous_attempt_error:
            prompt += f"\nPrevious attempt error: {previous_attempt_error}"
        prompt += "\nPlease provide the complete code for the SynthesizedInitialStateCommander and SynthesizedCommander classes."

        # Use the LLM to generate the Commander.
        response, _ = self._llm.query(
            prompt, temperature=self._llm_temperature, seed=llm_seed
        )
        synthesized_commander_code = parse_python_code_from_llm_response(response)

        # Add imports.
        synthesized_commander_code = (
            f"{combined_imports}\n\n{synthesized_commander_code}"
        )

        # Execute the synthesized code to create the Commander instance.
        try:
            exec(synthesized_commander_code, globals())  # pylint: disable=exec-used
            synthesized_commander = eval("SynthesizedCommander()")
        except Exception as e:
            print(f"WARNING: Failed to execute synthesized commander code. Error: {e}")
            return None, str(e)

        return synthesized_commander, ""
