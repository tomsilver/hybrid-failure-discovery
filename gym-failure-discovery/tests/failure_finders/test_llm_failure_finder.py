"""Tests for the LLM failure finder."""

import tempfile
from pathlib import Path
from typing import Any

import pytest
from tomsutils.llm import LargeLanguageModel, OpenAILLM

from gym_failure_discovery.envs.hovercraft_env import HoverCraftEnv, HoverCraftSceneSpec
from gym_failure_discovery.failure_finders.llm_failure_finder import LLMFailureFinder
from gym_failure_discovery.failure_monitors.hovercraft_failure_monitor import (
    HoverCraftFailureMonitor,
)


class _MockLLM(LargeLanguageModel):
    """An LLM that returns canned completions."""

    def __init__(self, completions: list[list[str]], cache_dir: Path) -> None:
        super().__init__(cache_dir, use_cache_only=False)
        self.completions = completions

    def get_id(self) -> str:
        return "mock"

    def _sample_completions(self, prompt, imgs, temperature, seed, num_completions=1):
        next_completions = self.completions.pop(0)
        assert num_completions == len(next_completions)
        return list(next_completions), {}

    def get_multiple_choice_logprobs(
        self, prompt: str, choices: list[str], seed: int
    ) -> tuple[dict[str, float], dict[str, Any]]:
        raise NotImplementedError


MOCK_POLICY_RESPONSE = """Here is a policy that should cause a failure:

```python
class SynthesizedPolicy:
    def reset(self):
        self._switched = False

    def act(self, obs):
        goal_x = 0.42
        current_x = obs[0]
        if abs(current_x - goal_x) < 0.1 and not self._switched:
            self._switched = True
            return 1
        return 0
```
"""


@pytest.mark.make_videos
def test_llm_failure_finder_with_mock(maybe_record):  # type: ignore
    """Mock LLM synthesizes a policy that causes a collision."""
    cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    llm = _MockLLM([[MOCK_POLICY_RESPONSE]], Path(cache_dir.name))
    spec = HoverCraftSceneSpec()
    env = maybe_record(HoverCraftEnv(spec))
    monitor = HoverCraftFailureMonitor(spec)
    finder = LLMFailureFinder(llm, seed=0, max_trajectory_length=500)
    result = finder.find_failure(env, monitor)
    assert result is not None
    assert len(result) > 0
    env.close()


@pytest.mark.run_llms
@pytest.mark.make_videos
def test_llm_failure_finder_openai(maybe_record):  # type: ignore
    """Run a real OpenAI LLM to find a hovercraft failure."""
    llm = OpenAILLM("gpt-4o", Path("./llm_cache"), max_tokens=4096)
    spec = HoverCraftSceneSpec()
    env = maybe_record(HoverCraftEnv(spec))
    monitor = HoverCraftFailureMonitor(spec)
    finder = LLMFailureFinder(
        llm, seed=0, max_trajectory_length=500, max_num_attempts=5
    )
    result = finder.find_failure(env, monitor)
    assert result is not None
    assert len(result) > 0
    env.close()
