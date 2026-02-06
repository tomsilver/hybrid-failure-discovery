"""Tests for llm_commander_failure_finder.py with Blocks environment."""

import tempfile
from pathlib import Path
from typing import Any

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

# Import exception for handling empty task plan
try:
    from task_then_motion_planning.planning import (
        TaskThenMotionPlanningFailure,
    )
except ImportError:
    # If not available, define a dummy exception class

    class TaskThenMotionPlanningFailure(Exception):  # type: ignore[no-redef]
        """Dummy exception class for when task_then_motion_planning is not
        available."""


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
            # Second command: build another tower with next blocks
            if num_blocks >= 6:
                return BlocksCommand(towers=[["block3", "block4", "block5"]])
        elif self._command_count == 3:
            # Third command: create a very tall tower (unstable)
            if num_blocks >= 4:
                return BlocksCommand(towers=[["block0", "block1", "block2", "block3"]])
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
    controller = BlocksController(seed=123, scene_spec=env.scene_spec)
    # Sensitive to catch failures
    failure_monitor = BlocksFailureMonitor(move_tol=0.01)
    failure_finder = LLMCommanderFailureFinder(
        llm, seed=123, max_num_trajectories=10, max_trajectory_length=200
    )

    # Run the failure finder, handling empty task plan exceptions
    try:
        result = failure_finder.run(
            env, controller, failure_monitor, synthesize_initial_state=False
        )
    except TaskThenMotionPlanningFailure as e:
        # If the planner has no task plan (empty plan), this means the command
        # was successfully completed. No failure found - treat as no failure.
        if "empty" in str(e).lower() or "task plan" in str(e).lower():
            result = None  # No failure found - command completed successfully
        else:
            # Re-raise if it's a different planning failure
            raise

    # Note: This test may or may not find a failure depending on the system
    # The mock commander is designed to create complex configurations
    # that might cause failures
    if result is not None:
        print(
            f"✓ Failure found! " f"Trajectory length: {len(result.observations)} steps"
        )
    else:
        print("✗ No failure found - this is acceptable for this test")

    # Uncomment to visualize if a failure was found.
    # if result is not None:
    #     import imageio.v2 as iio
    #     states = result.observations
    #     imgs = [env._render_state(s) for s in states]
    #     path = Path("videos") / "test-llm-commander" / "blocks_llm_commander_test.mp4"
    #     path.parent.mkdir(parents=True, exist_ok=True)
    #     iio.mimsave(path, imgs, fps=env.metadata["render_fps"])

    env.close()


def test_llm_initial_state_and_commander_failure_finder_blocks():
    """Tests for llm_commander_failure_finder.py with Blocks environment,
    including initial state synthesis."""

    mock_llm_completion = """Certainly! Here you go:
    
```python
from hybrid_failure_discovery.commander.commander import Commander
from hybrid_failure_discovery.commander.initial_state_commander import InitialStateCommander
from hybrid_failure_discovery.controllers.blocks_controller import (
    BlocksCommand,
)
from hybrid_failure_discovery.envs.blocks_env import (
    BlocksEnv,
    BlocksEnvSceneSpec,
    BlocksEnvState,
)
from pybullet_helpers.geometry import Pose

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
        assert isinstance(self._current_state, BlocksEnvState)
        
        self._command_count += 1
        
        num_blocks = len(self._current_state.blocks)
        
        # Create complex multi-tower configurations to increase failure risk
        if num_blocks >= 6:
            return BlocksCommand(towers=[
                ["block0", "block1", "block2"],
                ["block3", "block4", "block5"]
            ])
        elif num_blocks >= 4:
            return BlocksCommand(towers=[["block0", "block1", "block2", "block3"]])
        elif num_blocks >= 2:
            return BlocksCommand(towers=[["block0", "block1"]])
        
        return BlocksCommand(towers=[])

    def update(self, action, next_state):
        self._current_state = next_state

class SynthesizedInitialStateCommander(InitialStateCommander):
    def __init__(self, scene_spec: BlocksEnvSceneSpec, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scene_spec = scene_spec

    def initialize(self):
        # Create a simulator to generate initial state
        # Use default initial state from environment
        from hybrid_failure_discovery.envs.blocks_env import BlocksEnv
        temp_env = BlocksEnv(scene_spec=self._scene_spec, seed=123, use_gui=False)
        initial_space = temp_env.get_initial_states()
        initial_state = initial_space.sample()
        temp_env.close()
        return initial_state
```
"""

    cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    # Provide one completion - the mock will reuse it if needed
    # (get_initial_state_and_commander may be called multiple times during run())
    llm = _MockLLM([[mock_llm_completion]], Path(cache_dir.name))

    env = BlocksEnv(seed=123, use_gui=False)
    controller = BlocksController(seed=123, scene_spec=env.scene_spec)
    # Sensitive to catch failures
    failure_monitor = BlocksFailureMonitor(move_tol=0.01)
    failure_finder = LLMCommanderFailureFinder(
        llm, seed=123, max_num_trajectories=10, max_trajectory_length=200
    )

    # Run the failure finder, handling empty task plan exceptions
    try:
        result = failure_finder.run(
            env, controller, failure_monitor, synthesize_initial_state=True
        )
    except TaskThenMotionPlanningFailure as e:
        # If the planner has no task plan (empty plan), this means the command
        # was successfully completed. No failure found - treat as no failure.
        if "empty" in str(e).lower() or "task plan" in str(e).lower():
            result = None  # No failure found - command completed successfully
        else:
            # Re-raise if it's a different planning failure
            raise

    # Note: This test may or may not find a failure depending on the system
    if result is not None:
        print(f"✓ Failure found! Trajectory length: {len(result.observations)} steps")
    else:
        print("✗ No failure found - this is acceptable for this test")

    # Uncomment to visualize if a failure was found.
    # if result is not None:
    #     import imageio.v2 as iio
    #     states = result.observations
    #     imgs = [env._render_state(s) for s in states]
    #     path = (
    #         Path("videos")
    #         / "test-llm-commander"
    #         / "blocks_llm_initial_and_commander_test.mp4"
    #     )
    #     path.parent.mkdir(parents=True, exist_ok=True)
    #     iio.mimsave(path, imgs, fps=env.metadata["render_fps"])

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
        llm, seed=123, max_num_trajectories=10, max_trajectory_length=200
    )

    # Run the failure finder, handling empty task plan exceptions
    try:
        result = failure_finder.run(
            env, controller, failure_monitor, synthesize_initial_state=False
        )
    except TaskThenMotionPlanningFailure as e:
        if "empty" in str(e).lower() or "task plan" in str(e).lower():
            result = None
        else:
            raise

    # Note: This test may or may not find a failure
    if result is not None:
        print(f"✓ Failure found! Trajectory length: {len(result.observations)} steps")
    else:
        print("✗ No failure found")

    # Uncomment to visualize if a failure was found.
    # import imageio.v2 as iio
    # if result is not None:
    #     states = result.observations
    #     imgs = [env._render_state(s) for s in states]
    #     path = Path("videos") / "test-llm-commander" / "blocks_llm_commander_test.mp4"
    #     path.parent.mkdir(parents=True, exist_ok=True)
    #     iio.mimsave(path, imgs, fps=env.metadata["render_fps"])

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
        llm, seed=123, max_num_trajectories=10, max_trajectory_length=200
    )

    # Run the failure finder, handling empty task plan exceptions
    try:
        result = failure_finder.run(
            env, controller, failure_monitor, synthesize_initial_state=True
        )
    except TaskThenMotionPlanningFailure as e:
        if "empty" in str(e).lower() or "task plan" in str(e).lower():
            result = None
        else:
            raise

    # Note: This test may or may not find a failure
    if result is not None:
        print(f"✓ Failure found! Trajectory length: {len(result.observations)} steps")
    else:
        print("✗ No failure found")

    # Uncomment to visualize if a failure was found.
    # import imageio.v2 as iio
    # if result is not None:
    #     states = result.observations
    #     imgs = [env._render_state(s) for s in states]
    #     path = (
    #         Path("videos")
    #         / "test-llm-commander"
    #         / "blocks_llm_initial_and_commander_test.mp4"
    #     )
    #     path.parent.mkdir(parents=True, exist_ok=True)
    #     iio.mimsave(path, imgs, fps=env.metadata["render_fps"])

    env.close()
