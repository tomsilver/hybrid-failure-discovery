"""Tests for llm_commander_failure_finder.py."""

import tempfile
from pathlib import Path
from typing import Any

import pytest
from tomsutils.llm import LargeLanguageModel, OpenAILLM

from hybrid_failure_discovery.controllers.blocks_controller import BlocksController
from hybrid_failure_discovery.controllers.hovercraft_controller import (
    HoverCraftController,
)
from hybrid_failure_discovery.envs.blocks_env import (
    BlocksEnv,
)
from hybrid_failure_discovery.envs.hovercraft_env import (
    HoverCraftEnv,
)
from hybrid_failure_discovery.failure_finders.llm_commander_failure_finder import (
    LLMCommanderFailureFinder,
)
from hybrid_failure_discovery.failure_monitors.blocks_failure_monitor import (
    BlocksFailureMonitor,
)
from hybrid_failure_discovery.failure_monitors.hovercraft_failure_monitor import (
    HoverCraftFailureMonitor,
)


class _MockLLM(LargeLanguageModel):

    def __init__(
        self,
        completions: list[list[str]],
        cache_dir: Path,
        use_cache_only: bool = False,
    ) -> None:
        super().__init__(cache_dir, use_cache_only)
        self.completions = completions

    def get_id(self) -> str:
        return "mock"

    def _sample_completions(self, prompt, imgs, temperature, seed, num_completions=1):
        del imgs  # unused
        next_completions = self.completions.pop(0)
        assert num_completions == len(next_completions)
        return list(next_completions), {}

    def get_multiple_choice_logprobs(
        self, prompt: str, choices: list[str], seed: int
    ) -> tuple[dict[str, float], dict[str, Any]]:
        raise NotImplementedError("TODO")


def test_llm_commander_failure_finder():
    """Tests for llm_commander_failure_finder.py."""

    mock_llm_completion = """Certainly! Here you go:
    
```python
from hybrid_failure_discovery.commander.commander import Commander
from hybrid_failure_discovery.controllers.hovercraft_controller import (
    HoverCraftCommand,
    HoverCraftController,
)
from hybrid_failure_discovery.envs.hovercraft_env import (
    HoverCraftEnv,
    HoverCraftSceneSpec,
    HoverCraftState,
)

class SynthesizedCommander(Commander):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scene_spec = HoverCraftSceneSpec()
        self._current_state = None
        self._switched = False

    def reset(self, initial_state):
        self._current_state = initial_state

    def get_command(self):
        # Force a switch when near the right goal.
        assert isinstance(self._current_state, HoverCraftState)
        current_x = self._current_state.x
        goal_x = self._scene_spec.goal_pairs[0][1][0]
        dist = abs(current_x - goal_x)
        if dist < 0.1 and not self._switched:
            self._switched = True
            return HoverCraftCommand(switch=True)
        return HoverCraftCommand(switch=False)

    def update(self, action, next_state):
        self._current_state = next_state
```
"""

    cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    llm = _MockLLM([[mock_llm_completion]], Path(cache_dir.name))

    env = HoverCraftEnv()
    controller = HoverCraftController(123, env.scene_spec)
    failure_monitor = HoverCraftFailureMonitor(env.scene_spec)
    failure_finder = LLMCommanderFailureFinder(llm, seed=123)
    result = failure_finder.run(env, controller, failure_monitor)
    assert result is not None

    # Uncomment to visualize.
    # import imageio.v2 as iio
    # states = result.observations
    # imgs = [env._render_state(s) for s in states]
    # path = Path("videos") / "test-llm-commander" / "llm_commander_test.mp4"
    # path.parent.mkdir(exist_ok=True)
    # iio.mimsave(path, imgs, fps=env.metadata["render_fps"])


# @pytest.mark.skip(reason="Don't want to run actual LLM in CI.")
def test_openai_llm_hovercraft_commander_failure_finder():
    """Run an OpenAI LLM to create a failure finder commander for
    hovercraft."""

    llm = OpenAILLM("gpt-4o", Path("./llm_cache"), max_tokens=4096)

    env = HoverCraftEnv()
    controller = HoverCraftController(123, env.scene_spec)
    failure_monitor = HoverCraftFailureMonitor(env.scene_spec)
    failure_finder = LLMCommanderFailureFinder(
        llm, seed=123, max_num_trajectories=10, max_trajectory_length=50
    )
    result = failure_finder.run(env, controller, failure_monitor)
    assert result is not None

    # Uncomment to visualize.
    # import imageio.v2 as iio
    # states = result.observations
    # imgs = [env._render_state(s) for s in states]
    # path = Path("videos") / "test-llm-commander" / "hovercraft_llm_commander_test.mp4"
    # path.parent.mkdir(exist_ok=True)
    # iio.mimsave(path, imgs, fps=env.metadata["render_fps"])


# @pytest.mark.skip(reason="Don't want to run actual LLM in CI.")
def test_openai_llm_blocks_commander_failure_finder():
    """Run an OpenAI LLM to create a failure finder commander for blocks."""

    llm = OpenAILLM("gpt-4o", Path("./llm_cache"), max_tokens=4096)

    env = BlocksEnv()
    controller = BlocksController(123, env.scene_spec)
    failure_monitor = BlocksFailureMonitor()
    failure_finder = LLMCommanderFailureFinder(
        llm, seed=123, max_trajectory_length=1000
    )
    result = failure_finder.run(env, controller, failure_monitor)
    assert result is not None

    # Uncomment to visualize.
    import imageio.v2 as iio

    states = result.observations
    imgs = [env._render_state(s) for s in states]
    path = Path("videos") / "test-llm-commander" / "blocks_llm_commander_test.mp4"
    path.parent.mkdir(exist_ok=True)
    iio.mimsave(path, imgs, fps=env.metadata["render_fps"])
