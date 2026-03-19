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
    """Uses an LLM to synthesize a policy that induces a failure."""

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
            result = rollout(env, monitor, policy, seed, self._max_trajectory_length)
            if result is not None:
                return result
        return None

    def _synthesize_policy(
        self,
        env: gym.Env[ObsType, ActType],
        monitor: FailureMonitor,
        seed: int,
    ) -> Policy | None:
        env_source = _get_source(env)
        monitor_source = _get_source(monitor)

        prompt = f"""Given the following gymnasium environment and failure monitor:

Environment:
```python
{env_source}
```

Failure Monitor:
```python
{monitor_source}
```

The environment has action space: {env.action_space}
The environment has observation space: {env.observation_space}

A failure is detected by the monitor's `step` method returning True.

Please write a Python class called `SynthesizedPolicy` that has:
- a `reset(self)` method to initialize/reset any internal state
- an `act(self, obs)` method that takes an observation and returns an action

The policy should try to induce a failure (make the monitor's `step` return True).

```python
class SynthesizedPolicy:

    def reset(self):
        ...
    def act(self, obs):
        ...
```

You may use numpy (imported as np). Do NOT import anything else.
Provide only the class definition.
"""

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


def _get_source(obj: object) -> str:
    """Get the source code of the module defining obj's class."""
    module = inspect.getmodule(obj.__class__)
    assert module is not None
    return inspect.getsource(module)
