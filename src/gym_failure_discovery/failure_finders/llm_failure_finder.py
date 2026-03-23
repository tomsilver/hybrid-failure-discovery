"""A failure finder that uses an LLM to synthesize a policy."""

import inspect
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from tomsutils.llm import LargeLanguageModel, parse_python_code_from_llm_response
from tomsutils.utils import sample_seed_from_rng

from gym_failure_discovery.failure_finders.failure_finder import FailureFinder
from gym_failure_discovery.failure_monitors.failure_monitor import FailureMonitor
from gym_failure_discovery.utils import Policy, rollout


class LLMFailureFinder(FailureFinder):
    """Uses an LLM to synthesize a policy that induces a failure.

    The LLM is given the full source of the failure monitor (so it knows the
    exact failure condition) but only the docstrings of the environment module
    (no implementation details).  This prevents it from reading internal state
    variables or logic to trivially engineer a failure.
    """

    def __init__(
        self,
        llm: LargeLanguageModel,
        seed: int = 0,
        max_trajectory_length: int = 500,
        max_num_attempts: int = 5,
        llm_temperature: float = 0.7,
        num_synthesis_retries: int = 3,
    ) -> None:
        self._llm = llm
        self._rng = np.random.default_rng(seed)
        self._max_trajectory_length = max_trajectory_length
        self._max_num_attempts = max_num_attempts
        self._llm_temperature = llm_temperature
        self._num_synthesis_retries = num_synthesis_retries

    def find_failure(
        self,
        env: gym.Env[ObsType, ActType],
        monitor: FailureMonitor,
    ) -> list[tuple[ObsType, ActType]] | None:
        for _ in range(self._max_num_attempts):
            seed = sample_seed_from_rng(self._rng)
            policy = self._synthesize_policy(env, monitor, seed)
            if policy is None:
                continue
            try:
                result = rollout(
                    env, monitor, policy, seed, self._max_trajectory_length
                )
            except Exception:  # pylint: disable=broad-except
                continue
            if result is not None:
                return result
        return None

    def _synthesize_policy(
        self,
        env: gym.Env[ObsType, ActType],
        monitor: FailureMonitor,
        seed: int,
    ) -> Policy | None:
        env_block = _get_docstring(env)
        monitor_block = f"```python\n{_get_source(monitor)}\n```"

        prompt = (
            "You are searching for inputs that cause a gymnasium"
            " environment to fail.\n\n"
            f"Environment context:\n{env_block}\n\n"
            f"Failure monitor context:\n{monitor_block}\n\n"
            f"The environment has action space: {env.action_space}\n"
            "The environment has observation space:"
            f" {env.observation_space}\n\n"
            "A failure is detected by the monitor's `step` method"
            " returning True.\n\n"
            "Please write a Python class called `SynthesizedPolicy`"
            " that has:\n"
            "- a `reset(self)` method to initialize/reset any"
            " internal state\n"
            "- an `act(self, obs)` method that takes an observation"
            " and returns an action\n\n"
            "The policy should try to induce a failure (make the"
            " monitor's `step` return True).\n\n"
            "```python\n"
            "class SynthesizedPolicy:\n\n"
            "    def reset(self):\n"
            "        ...\n"
            "    def act(self, obs):\n"
            "        ...\n"
            "```\n\n"
            "You may use numpy (imported as np)."
            " Do NOT import anything else.\n"
            "Provide only the class definition.\n"
        )

        previous_error: str | None = None
        for _ in range(self._num_synthesis_retries):
            full_prompt = prompt
            if previous_error:
                full_prompt += f"\nPrevious attempt failed with: {previous_error}"
            response, _ = self._llm.query(
                full_prompt,
                temperature=self._llm_temperature,
                seed=seed,
            )
            code = parse_python_code_from_llm_response(response)
            namespace: dict = {"np": np}
            try:
                exec(code, namespace)  # pylint: disable=exec-used
                instance = namespace["SynthesizedPolicy"]()
                return _WrappedPolicy(instance)
            except Exception as exc:  # pylint: disable=broad-except
                previous_error = str(exc)
        return None


class _WrappedPolicy(Policy):
    """Wraps a dynamically created policy object as a Policy."""

    def __init__(self, inner: object) -> None:
        self._inner = inner

    def reset(self) -> None:
        self._inner.reset()  # type: ignore[attr-defined]

    def act(self, obs: Any) -> Any:
        return self._inner.act(obs)  # type: ignore[attr-defined]


def _actual(obj: object) -> object:
    """Unwrap gymnasium wrappers to reach the real environment/monitor."""
    return obj.unwrapped if isinstance(obj, gym.Wrapper) else obj


def _get_source(obj: object) -> str:
    """Return the full source of the module that defines obj's class."""
    module = inspect.getmodule(_actual(obj).__class__)
    assert module is not None
    return inspect.getsource(module)


def _get_docstring(obj: object) -> str:
    """Return a concise description built from module- and class-level docstrings."""
    cls = _actual(obj).__class__
    module = inspect.getmodule(cls)
    assert module is not None
    parts: list[str] = []
    if module.__doc__:
        parts.append(f"Module description:\n{inspect.cleandoc(module.__doc__)}")
    class_doc = inspect.getdoc(cls)
    if class_doc:
        parts.append(f"Class `{cls.__name__}` description:\n{class_doc}")
    return (
        "\n\n".join(parts) if parts else f"(No docstring available for {cls.__name__})"
    )
