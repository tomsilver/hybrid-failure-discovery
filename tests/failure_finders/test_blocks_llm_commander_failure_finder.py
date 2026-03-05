"""Tests for llm_commander_failure_finder.py with Blocks environment."""

import tempfile
from pathlib import Path
from typing import Any

import imageio.v2 as iio
import pytest
from tomsutils.llm import LargeLanguageModel

from hybrid_failure_discovery.controllers.blocks_controller import BlocksController
from hybrid_failure_discovery.envs.blocks_env import BlocksEnv
from hybrid_failure_discovery.failure_finders.llm_commander_failure_finder import (
    LLMCommanderFailureFinder,
)
from hybrid_failure_discovery.failure_monitors.blocks_failure_monitor import (
    BlocksFailureMonitor,
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
        self._last_completions: list[str] | None = None

    def get_id(self) -> str:
        return "mock"

    def _sample_completions(self, prompt, imgs, temperature, seed, num_completions=1):
        del imgs  # unused
        # Reuse the last completion if we run out
        # (for cases where synthesis is called multiple times)
        if len(self.completions) == 0:
            # If we've exhausted all completions, reuse the last one we had
            # This handles cases where get_initial_state_and_commander
            # is called multiple times
            if self._last_completions is None:
                raise IndexError(
                    "No completions available and no previous completion to reuse"
                )
            next_completions = self._last_completions
        else:
            next_completions = self.completions.pop(0)
            self._last_completions = next_completions  # Store for reuse
        assert num_completions == len(next_completions)
        return list(next_completions), {}

    def get_multiple_choice_logprobs(
        self, prompt: str, choices: list[str], seed: int
    ) -> tuple[dict[str, float], dict[str, Any]]:
        raise NotImplementedError("TODO")


def test_llm_commander_failure_finder_blocks():
    """Tests for llm_commander_failure_finder.py with Blocks environment."""

    mock_llm_completion = """Certainly! Here you go:
    
```python
from hybrid_failure_discovery.commander.commander import Commander
from hybrid_failure_discovery.controllers.blocks_controller import (
    BlocksCommand,
)
from hybrid_failure_discovery.envs.blocks_env import (
    BlocksEnvState,
)

class SynthesizedCommander(Commander):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_state = None
        self._command_count = 0

    def reset(self, initial_state):
        self._current_state = initial_state
        self._command_count = 0

    def get_command(self):
        # Generate commands that create complex tower configurations
        # This should increase the likelihood of failures (collisions, knock-overs)
        assert isinstance(self._current_state, BlocksEnvState)
        
        self._command_count += 1
        
        # Create increasingly complex tower configurations
        # Start with simple towers, then build more complex ones
        num_blocks = len(self._current_state.blocks)
        
        if self._command_count == 1:
            # First command: build a simple tower with first 3 blocks
            if num_blocks >= 3:
                return BlocksCommand(towers=[["block0", "block1", "block2"]])
        elif self._command_count == 2:
            # Second command: create a very tall tower (unstable)
            if num_blocks >= 4:
                return BlocksCommand(towers=[["block1", "block0", "block2", "block3"]])
        else:
            # Subsequent commands: create complex multi-tower configurations
            # This should increase collision risk
            if num_blocks >= 6:
                return BlocksCommand(towers=[
                    ["block0", "block1"],
                    ["block2", "block3"],
                    ["block4", "block5"]
                ])
        
        # Default: simple tower
        if num_blocks >= 2:
            return BlocksCommand(towers=[["block0", "block1"]])
        return BlocksCommand(towers=[])

    def update(self, action, next_state):
        self._current_state = next_state
```
"""

    cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    # Provide one completion - the mock will reuse it if needed
    # (get_initial_state_and_commander may be called multiple times during run())
    llm = _MockLLM([[mock_llm_completion]], Path(cache_dir.name))

    env = BlocksEnv(seed=123, use_gui=False)
    controller = BlocksController(seed=123, scene_spec=env.scene_spec, safe_height=0.2)
    # Sensitive to catch failures
    failure_monitor = BlocksFailureMonitor(move_tol=0.01)
    failure_finder = LLMCommanderFailureFinder(
        llm, seed=123, max_num_trajectories=1, max_trajectory_length=10000
    )

    # Run the failure finder, handling empty task plan exceptions
    # try:
    result = failure_finder.run(
        env, controller, failure_monitor, synthesize_initial_state=False
    )

    if result is not None:
        print(f"✓ Failure found! Trajectory length: {len(result.observations)} steps")
        reason = failure_monitor.failure_reason or "unknown"
        print(f"  Failure cause: {reason}")
    else:
        print("✗ No failure found - this is acceptable for this test")

    # traj_to_render = result or failure_finder.last_trajectory
    # if traj_to_render is not None:
    #     states = traj_to_render.observations
    # else:
    #     initial_state = env.get_initial_states().sample()
    #     states = [initial_state] * 30
    # # pylint: disable=protected-access
    # imgs = [env._render_state(s) for s in states]
    # path = Path("videos") / "test-llm-commander" / "blocks_llm_commander_test.mp4"
    # path.parent.mkdir(parents=True, exist_ok=True)
    # iio.mimsave(path, imgs, fps=env.metadata["render_fps"])

    env.close()


@pytest.mark.skip(reason="Don't want to run actual LLM in CI.")
def test_openai_llm_blocks_commander_failure_finder():
    """Run an OpenAI LLM to create a failure finder commander for blocks."""

    from tomsutils.llm import OpenAILLM  # pylint: disable=import-outside-toplevel

    llm = OpenAILLM("gpt-4o", Path("./llm_cache"), max_tokens=4096)

    env = BlocksEnv(seed=123, use_gui=False)
    controller = BlocksController(seed=123, scene_spec=env.scene_spec)
    failure_monitor = BlocksFailureMonitor(move_tol=0.01)
    failure_finder = LLMCommanderFailureFinder(
        llm, seed=123, max_num_trajectories=50, max_trajectory_length=10000
    )

    # Run the failure finder, handling empty task plan exceptions
    # try:
    result = failure_finder.run(
        env, controller, failure_monitor, synthesize_initial_state=False
    )
    # Note: This test may or may not find a failure
    if result is not None:
        print(f"✓ Failure found! Trajectory length: {len(result.observations)} steps")
    else:
        print("✗ No failure found")

    # traj_to_render = result or failure_finder.last_trajectory
    # if traj_to_render is not None:
    #     states = traj_to_render.observations
    # else:
    #     initial_state = env.get_initial_states().sample()
    #     states = [initial_state] * 30
    # # pylint: disable=protected-access
    # imgs = [env._render_state(s) for s in states]
    # path = Path("videos") / "test-llm-commander" / "blocks_llm_commander_test.mp4"
    # path.parent.mkdir(parents=True, exist_ok=True)
    # iio.mimsave(path, imgs, fps=env.metadata["render_fps"])

    env.close()


@pytest.mark.skip(reason="Don't want to run actual LLM in CI.")
def test_openai_llm_blocks_initial_state_and_commander_failure_finder():
    """Run an OpenAI LLM to create a failure finder commander for blocks,
    including initial state synthesis."""

    from tomsutils.llm import OpenAILLM  # pylint: disable=import-outside-toplevel

    llm = OpenAILLM("gpt-4o", Path("./llm_cache"), max_tokens=4096)

    env = BlocksEnv(seed=123, use_gui=False)
    controller = BlocksController(seed=123, scene_spec=env.scene_spec)
    failure_monitor = BlocksFailureMonitor(move_tol=0.01)
    failure_finder = LLMCommanderFailureFinder(
        llm, seed=123, max_num_trajectories=50, max_trajectory_length=10000
    )

    # Run the failure finder, handling empty task plan exceptions
    # try:
    result = failure_finder.run(
        env, controller, failure_monitor, synthesize_initial_state=True
    )

    # Note: This test may or may not find a failure
    if result is not None:
        print(f"✓ Failure found! Trajectory length: {len(result.observations)} steps")
    else:
        print("✗ No failure found")

    traj_to_render = result or failure_finder.last_trajectory
    if traj_to_render is not None:
        states = traj_to_render.observations
    else:
        initial_state = env.get_initial_states().sample()
        states = [initial_state] * 30
    # pylint: disable=protected-access
    imgs = [env._render_state(s) for s in states]
    path = (
        Path("videos")
        / "test-llm-commander"
        / "blocks_llm_initial_and_commander_test.mp4"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.mimsave(path, imgs, fps=env.metadata["render_fps"])

    env.close()
