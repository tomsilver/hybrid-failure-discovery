"""A failure finder that synthesizes a commander using an LLM."""

import ast
import inspect
import re
from typing import Any, Optional

from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Space
from tomsutils.llm import LargeLanguageModel, parse_python_code_from_llm_response
from tomsutils.utils import sample_seed_from_rng

from hybrid_failure_discovery.commander.commander import Commander
from hybrid_failure_discovery.commander.initial_state_commander import (
    InitialStateCommander,
)
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
from hybrid_failure_discovery.structs import CommandType, Trajectory
from hybrid_failure_discovery.utils import extend_trajectory_until_failure


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
        self._commander: Optional[Commander[ObsType, ActType, CommandType]] = None
        self._initial_state_commander: Optional[InitialStateCommander[ObsType]] = None
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

        # previous_attempt_error: str | None = None
        # for _ in range(self._num_synthesis_retries):
        #     llm_seed = sample_seed_from_rng(self._rng)
        #     commander, previous_attempt_error = self._synthesize_commander_with_llm(
        #         env, controller, failure_monitor, llm_seed, previous_attempt_error
        #     )
        #     if commander is not None:
        #         return commander
        # raise RuntimeError("Failed to synthesize a commander with the LLM.")
        assert self._commander is not None, (
            "Commander has not been synthesized yet. "
            "Call get_initial_state_and_commander() first."
        )
        return self._commander

    def get_initial_state(
        self,
        initial_space: Space[ObsType],
        env: ConstraintBasedEnvModel[ObsType, ActType],
        controller: ConstraintBasedController[ObsType, ActType, CommandType],
        failure_monitor: FailureMonitor[ObsType, ActType, CommandType],
    ) -> InitialStateCommander:
        # seed = sample_seed_from_rng(self._rng)
        # initializer = RandomInitialStateCommander(initial_space)
        # initializer.seed(seed)
        assert self._initial_state_commander is not None, (
            "Initial state commander has not been synthesized yet. "
            "Call get_initial_state_and_commander() first."
        )
        return self._initial_state_commander

    def get_initial_state_and_commander(
        self,
        initial_space: Space[ObsType],
        env: ConstraintBasedEnvModel[ObsType, ActType],
        controller: ConstraintBasedController[ObsType, ActType, CommandType],
        failure_monitor: FailureMonitor[ObsType, ActType, CommandType],
        synthesize_initial_state: bool = True,
    ):
        """Queries LLM to synthesize an initial state and a goal commander."""
        previous_attempt_error: str | None = None
        llm_seed = sample_seed_from_rng(self._rng)
        if synthesize_initial_state:
            for _ in range(self._num_synthesis_retries):

                (
                    self._initial_state_commander,
                    self._commander,
                    previous_attempt_error,
                ) = self._synthesize_initial_state_and_commander_with_llm(
                    env,
                    controller,
                    failure_monitor,
                    llm_seed,
                    previous_attempt_error,
                )
                if (
                    self._initial_state_commander is not None
                    and self._commander is not None
                ):
                    break
        else:
            initial_state_commander_seed = sample_seed_from_rng(self._rng)
            self._initial_state_commander = RandomInitialStateCommander(initial_space)
            self._initial_state_commander.seed(initial_state_commander_seed)

            for _ in range(self._num_synthesis_retries):
                self._commander, previous_attempt_error = (
                    self._synthesize_commander_with_llm(
                        env,
                        controller,
                        failure_monitor,
                        llm_seed,
                        previous_attempt_error,
                    )
                )
                if self._commander is not None:
                    break
        if self._initial_state_commander is None or self._commander is None:
            raise RuntimeError("Failed to synthesize commanders with the LLM.")

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

        # Combine all import statements.
        combined_imports = "from __future__ import annotations\n"
        combined_imports += "\n".join(
            set(
                env_imports.split("\n")
                + controller_imports.split("\n")
                + failure_monitor_imports.split("\n")
                + commander_imports.split("\n")
            )
        )

        # Determine the module names of the env, controller, and failure_monitor.
        env_module = env.__class__.__module__
        controller_module = controller.__class__.__module__
        failure_monitor_module = failure_monitor.__class__.__module__
        commander_module = Commander.__module__

        # Generate import statements to import everything from those modules.
        additional_imports = f"from {env_module} import *\nfrom {controller_module} import *\nfrom {failure_monitor_module} import *\nfrom {commander_module} import Commander"  # pylint: disable=line-too-long
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

Please synthesize a Commander that would induce a failure.

Your class should be called SynthesizedCommander() and should take no arguments in the constructor.
"""
        if previous_attempt_error:
            prompt += f"\nPrevious attempt error: {previous_attempt_error}"
        prompt += (
            "\nPlease provide the complete code for the SynthesizedCommander class."
        )

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

    def _synthesize_initial_state_and_commander_with_llm(
        self,
        env: ConstraintBasedEnvModel[ObsType, ActType],
        controller: ConstraintBasedController[ObsType, ActType, CommandType],
        failure_monitor: FailureMonitor[ObsType, ActType, CommandType],
        llm_seed: int,
        previous_attempt_error: str | None = None,
    ) -> tuple[
        InitialStateCommander | None,
        Commander[ObsType, ActType, CommandType] | None,
        str,
    ]:
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
        initial_state_commander_imports = _extract_imports(
            initial_state_commander_source
        )

        # Combine all import statements.
        combined_imports = "from __future__ import annotations\n"
        combined_imports += "\n".join(
            set(
                env_imports.split("\n")
                + controller_imports.split("\n")
                + failure_monitor_imports.split("\n")
                + commander_imports.split("\n")
                + initial_state_commander_imports.split("\n")
            )
        )

        # Determine the module names of the env, controller, and failure_monitor.
        env_module = env.__class__.__module__
        controller_module = controller.__class__.__module__
        failure_monitor_module = failure_monitor.__class__.__module__
        commander_module = Commander.__module__
        initial_state_commander_module = InitialStateCommander.__module__

        # Generate import statements to import everything from those modules.
        additional_imports = f"from {env_module} import *\nfrom {controller_module} import *\nfrom {failure_monitor_module} import *\nfrom {commander_module} import Commander"  # pylint: disable=line-too-long
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
{initial_state_commander_source}

Please synthesize an InitialStateCommander and a Commander that would induce a failure.

Your classes should be called SynthesizedInitialStateCommander() and SynthesizedCommander(). The SynthesizedInitialStateCommander() should take one argument in the constructor: initial_space, which is the output of env.get_initial_states(). The SynthesizedCommander() should take no arguments in the constructor.
"""
        if previous_attempt_error:
            prompt += f"\nPrevious attempt error: {previous_attempt_error}"
        prompt += (
            "\nPlease provide the complete code for the SynthesizedCommander class."
        )

        # Use the LLM to generate the Commander.
        response, _ = self._llm.query(
            prompt, temperature=self._llm_temperature, seed=llm_seed
        )
        synthesized_initial_state_commander_code: str
        synthesized_commander_code: str
        synthesized_commanders = parse_python_code_from_llm_response(response)

        # Use regex to split on the second class definition
        matches = list(
            re.finditer(
                r"^class\s+SynthesizedCommander\b",
                synthesized_commanders,
                flags=re.MULTILINE,
            )
        )

        if not matches:
            raise RuntimeError("Could not find SynthesizedCommander class in the code")

        split_index = matches[0].start()

        synthesized_initial_state_commander_code = synthesized_commanders[
            :split_index
        ].strip()
        synthesized_commander_code = synthesized_commanders[split_index:].strip()

        # Add imports.
        synthesized_initial_state_commander_code = (
            f"{combined_imports}\n\n{synthesized_initial_state_commander_code}"
        )
        synthesized_commander_code = (
            f"{combined_imports}\n\n{synthesized_commander_code}"
        )

        # Execute the synthesized code to create the Commander instance.
        try:
            exec(synthesized_commander_code, globals())  # pylint: disable=exec-used
            synthesized_commander = eval("SynthesizedCommander()")

            exec(
                synthesized_initial_state_commander_code, globals()
            )  # pylint: disable=exec-used
            synthesized_initial_state_commander = eval(
                "SynthesizedInitialStateCommander()"
            )
        except Exception as e:
            print(f"WARNING: Failed to execute synthesized commander code. Error: {e}")
            return None, None, str(e)

        return synthesized_initial_state_commander, synthesized_commander, ""

    def run(
        self,
        env: ConstraintBasedEnvModel[ObsType, ActType],
        controller: ConstraintBasedController[ObsType, ActType, CommandType],
        failure_monitor: FailureMonitor[ObsType, ActType, CommandType],
        synthesize_initial_state: bool = True,
    ) -> Trajectory[ObsType, ActType, CommandType] | None:
        for traj_idx in range(self._max_num_trajectories):
            # Initialize the particles (partial trajectories).
            initial_space = env.get_initial_states()
            self.get_initial_state_and_commander(
                initial_space,
                env,
                controller,
                failure_monitor,
                synthesize_initial_state,
            )
            initializer = self.get_initial_state(
                initial_space, env, controller, failure_monitor
            )
            initial_state = initializer.initialize()

            init_traj: Trajectory[ObsType, ActType, CommandType] = Trajectory(
                [initial_state], [], []
            )

            commander = self.get_commander(env, controller, failure_monitor, traj_idx)

            def _termination_fn(traj: Trajectory) -> bool:
                return len(traj.actions) >= self._max_trajectory_length

            failure_traj, failure_found = extend_trajectory_until_failure(
                init_traj,
                env,
                commander,
                controller,
                failure_monitor,
                _termination_fn,
                self._rng,
            )

            # Failure found, we're done!
            if failure_found:
                print(f"Found a failure after {traj_idx+1} trajectory samples")
                return failure_traj
        print("Failure finding failed.")
        return None
