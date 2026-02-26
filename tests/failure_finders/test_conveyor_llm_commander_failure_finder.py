"""Tests for llm_commander_failure_finder.py with ConveyorBelt environment."""

import tempfile
from pathlib import Path
from typing import Any

import pytest
from tomsutils.llm import LargeLanguageModel

from hybrid_failure_discovery.controllers.conveyorbelt_controller import (
    ConveyorBeltController,
)
from hybrid_failure_discovery.envs.conveyorbelt_env import (
    ConveyorBeltEnv,
    ConveyorBeltSceneSpec,
)
from hybrid_failure_discovery.failure_finders.llm_commander_failure_finder import (
    LLMCommanderFailureFinder,
)
from hybrid_failure_discovery.failure_monitors.conveyorbelt_failure_monitor import (
    ConveyorBeltFailureMonitor,
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


def test_llm_commander_failure_finder_conveyorbelt():
    """Tests for llm_commander_failure_finder.py with ConveyorBelt environment.

    The mock commander generates the secret failure mode sequence:
    ["fast", "mid", "fast", "slow", "off", "slow", "slow", "fast"]
    """

    mock_llm_completion = """Certainly! Here you go:
    
```python
from hybrid_failure_discovery.commander.commander import Commander
from hybrid_failure_discovery.controllers.conveyorbelt_controller import (
    ConveyorBeltCommand,
)
from hybrid_failure_discovery.envs.conveyorbelt_env import (
    ConveyorBeltState,
)

class SynthesizedCommander(Commander):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_state = None
        self._command_count = 0
        # Secret failure sequence:
        # ["fast", "mid", "fast", "slow", "off", "slow", "slow", "fast"]
        self._secret_sequence = [
            "fast",
            "mid",
            "fast",
            "slow",
            "off",
            "slow",
            "slow",
            "fast",
        ]

    def reset(self, initial_state):
        self._current_state = initial_state
        self._command_count = 0

    def get_command(self):
        # Generate commands following the secret failure sequence
        assert isinstance(self._current_state, ConveyorBeltState)
        
        # Cycle through the secret sequence
        mode_index = self._command_count % len(self._secret_sequence)
        mode = self._secret_sequence[mode_index]
        
        self._command_count += 1
        
        return ConveyorBeltCommand(mode=mode)

    def update(self, action, next_state):
        self._current_state = next_state
```
"""

    cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    # Provide one completion - the mock will reuse it if needed
    # (get_initial_state_and_commander may be called multiple times)
    llm = _MockLLM([[mock_llm_completion]], Path(cache_dir.name))

    # Test failure finder in conveyorbelt env
    # Create a scene spec with moderate constraints
    scene_spec = ConveyorBeltSceneSpec(
        box_width=0.3,
        conveyor_belt_velocity=2.0,
        dt=0.01,
        belt_length=3.0,
    )
    object.__setattr__(scene_spec, "min_spacing", 0.1)

    env = ConveyorBeltEnv(scene_spec=scene_spec)
    secret_mode_sequence = [
        "fast",
        "mid",
        "fast",
        "slow",
        "off",
        "slow",
        "slow",
        "fast",
    ]
    controller = ConveyorBeltController(
        seed=123,
        scene_spec=env.scene_spec,
        secret_failure_mode_sequence=secret_mode_sequence,
    )
    failure_monitor = ConveyorBeltFailureMonitor(env.scene_spec)
    failure_finder = LLMCommanderFailureFinder(
        llm, seed=123, max_num_trajectories=1000, max_trajectory_length=200
    )

    result = failure_finder.run(
        env, controller, failure_monitor, synthesize_initial_state=False
    )

    # Assert that a failure was found
    # The commander should generate the secret sequence
    assert result is not None

    # Uncomment to visualize.
    # import imageio.v2 as iio
    # states = result.observations
    # imgs = [env._render_state(s) for s in states]
    # path = (
    #     Path("videos")
    #     / "test-llm-commander"
    #     / "conveyorbelt_llm_commander_test.mp4"
    # )
    # path.parent.mkdir(parents=True, exist_ok=True)
    # iio.mimsave(path, imgs, fps=env.metadata["render_fps"])


def test_llm_initial_state_and_commander_failure_finder_conveyorbelt():
    """Tests for llm_commander_failure_finder.py with ConveyorBelt environment,
    including initial state synthesis."""

    mock_llm_completion = """Certainly! Here you go:
    
```python
from hybrid_failure_discovery.commander.commander import Commander
from hybrid_failure_discovery.commander.initial_state_commander import (
    InitialStateCommander,
)
from hybrid_failure_discovery.controllers.conveyorbelt_controller import (
    ConveyorBeltCommand,
)
from hybrid_failure_discovery.envs.conveyorbelt_env import (
    ConveyorBeltEnv,
    ConveyorBeltSceneSpec,
    ConveyorBeltState,
)

class SynthesizedCommander(Commander):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_state = None
        self._command_count = 0
        # Secret failure sequence:
        # ["fast", "mid", "fast", "slow", "off", "slow", "slow", "fast"]
        self._secret_sequence = [
            "fast",
            "mid",
            "fast",
            "slow",
            "off",
            "slow",
            "slow",
            "fast",
        ]

    def reset(self, initial_state):
        self._current_state = initial_state
        self._command_count = 0

    def get_command(self):
        # Generate commands following the secret failure sequence
        assert isinstance(self._current_state, ConveyorBeltState)
        
        # Cycle through the secret sequence
        mode_index = self._command_count % len(self._secret_sequence)
        mode = self._secret_sequence[mode_index]
        
        self._command_count += 1
        
        return ConveyorBeltCommand(mode=mode)

    def update(self, action, next_state):
        self._current_state = next_state

class SynthesizedInitialStateCommander(InitialStateCommander):
    def __init__(self, scene_spec: ConveyorBeltSceneSpec, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scene_spec = scene_spec

    def initialize(self):
        # Use default initial state (empty conveyor belt)
        return ConveyorBeltState(
            positions=[],
            falling_heights=[],
            step_count=0,
        )
```
"""

    cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    # Provide one completion - the mock will reuse it if needed
    llm = _MockLLM([[mock_llm_completion]], Path(cache_dir.name))

    # Test failure finder in conveyorbelt env
    # Create a scene spec with moderate constraints
    scene_spec = ConveyorBeltSceneSpec(
        box_width=0.3,
        conveyor_belt_velocity=2.0,
        dt=0.01,
        belt_length=3.0,
    )
    object.__setattr__(scene_spec, "min_spacing", 0.1)

    env = ConveyorBeltEnv(scene_spec=scene_spec)
    secret_mode_sequence = [
        "fast",
        "mid",
        "fast",
        "slow",
        "off",
        "slow",
        "slow",
        "fast",
    ]
    controller = ConveyorBeltController(
        seed=123,
        scene_spec=env.scene_spec,
        secret_failure_mode_sequence=secret_mode_sequence,
    )
    failure_monitor = ConveyorBeltFailureMonitor(env.scene_spec)
    failure_finder = LLMCommanderFailureFinder(
        llm, seed=123, max_num_trajectories=10, max_trajectory_length=200
    )

    result = failure_finder.run(
        env, controller, failure_monitor, synthesize_initial_state=True
    )

    # Assert that a failure was found
    assert result is not None

    # Uncomment to visualize.
    # import imageio.v2 as iio
    # states = result.observations
    # imgs = [env._render_state(s) for s in states]
    # path = (
    #     Path("videos")
    #     / "test-llm-commander"
    #     / "conveyorbelt_llm_initial_and_commander_test.mp4"
    # )
    # path.parent.mkdir(parents=True, exist_ok=True)
    # iio.mimsave(path, imgs, fps=env.metadata["render_fps"])


@pytest.mark.skip(reason="Don't want to run actual LLM in CI.")
def test_openai_llm_conveyorbelt_commander_failure_finder():
    """Run an OpenAI LLM to create a failure finder commander for
    conveyorbelt."""

    from tomsutils.llm import OpenAILLM  # pylint: disable=import-outside-toplevel

    llm = OpenAILLM("gpt-4o", Path("./llm_cache"), max_tokens=4096)

    # Test failure finder in conveyorbelt env
    scene_spec = ConveyorBeltSceneSpec(
        box_width=0.3,
        conveyor_belt_velocity=2.0,
        dt=0.01,
        belt_length=3.0,
    )
    object.__setattr__(scene_spec, "min_spacing", 0.1)

    env = ConveyorBeltEnv(scene_spec=scene_spec)
    secret_mode_sequence = [
        "fast",
        "mid",
        "fast",
        "slow",
        "off",
        "slow",
        "slow",
        "fast",
    ]
    controller = ConveyorBeltController(
        seed=123,
        scene_spec=env.scene_spec,
        secret_failure_mode_sequence=secret_mode_sequence,
    )
    failure_monitor = ConveyorBeltFailureMonitor(env.scene_spec)
    failure_finder = LLMCommanderFailureFinder(
        llm, seed=123, max_num_trajectories=10, max_trajectory_length=200
    )

    result = failure_finder.run(
        env, controller, failure_monitor, synthesize_initial_state=False
    )

    # Note: This test may or may not find a failure depending on LLM output
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
    #         / "conveyorbelt_llm_commander_test.mp4"
    #     )
    #     path.parent.mkdir(parents=True, exist_ok=True)
    #     iio.mimsave(path, imgs, fps=env.metadata["render_fps"])


@pytest.mark.skip(reason="Don't want to run actual LLM in CI.")
def test_openai_llm_conveyorbelt_initial_state_and_commander_failure_finder():
    """Run an OpenAI LLM to create a failure finder commander for conveyorbelt,
    including initial state synthesis."""

    from tomsutils.llm import OpenAILLM  # pylint: disable=import-outside-toplevel

    llm = OpenAILLM("gpt-4o", Path("./llm_cache"), max_tokens=4096)

    # Test failure finder in conveyorbelt env
    scene_spec = ConveyorBeltSceneSpec(
        box_width=0.3,
        conveyor_belt_velocity=2.0,
        dt=0.01,
        belt_length=3.0,
    )
    object.__setattr__(scene_spec, "min_spacing", 0.1)

    env = ConveyorBeltEnv(scene_spec=scene_spec)
    secret_mode_sequence = [
        "fast",
        "mid",
        "fast",
        "slow",
        "off",
        "slow",
        "slow",
        "fast",
    ]
    controller = ConveyorBeltController(
        seed=123,
        scene_spec=env.scene_spec,
        secret_failure_mode_sequence=secret_mode_sequence,
    )
    failure_monitor = ConveyorBeltFailureMonitor(env.scene_spec)
    failure_finder = LLMCommanderFailureFinder(
        llm, seed=123, max_num_trajectories=10, max_trajectory_length=200
    )

    result = failure_finder.run(
        env, controller, failure_monitor, synthesize_initial_state=True
    )

    # Note: This test may or may not find a failure depending on LLM output
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
    #         / "conveyorbelt_llm_initial_and_commander_test.mp4"
    #     )
    #     path.parent.mkdir(parents=True, exist_ok=True)
    #     iio.mimsave(path, imgs, fps=env.metadata["render_fps"])
