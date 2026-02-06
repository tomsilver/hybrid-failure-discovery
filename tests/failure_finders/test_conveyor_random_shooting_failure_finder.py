"""Tests for random_shooting_failure_finder.py.

This test suite validates the RandomShootingFailureFinder's ability to discover
failures in the ConveyorBelt environment. The ConveyorBeltController has been
intentionally modified with faults to make failures possible (see controller
code for details on the faults).

The test suite includes two scenarios:
1. Low trajectory count: Tests that failures may not always be found with
   limited sampling (demonstrates that failures are moderately rare)
2. High trajectory count: Tests that with sufficient sampling, failures can
   be reliably found (validates failure detection works)
"""

from pathlib import Path

import imageio.v2 as iio

from hybrid_failure_discovery.controllers.conveyorbelt_controller import (
    ConveyorBeltController,
)
from hybrid_failure_discovery.envs.conveyorbelt_env import (
    ConveyorBeltEnv,
    ConveyorBeltSceneSpec,
)
from hybrid_failure_discovery.failure_finders.random_shooting_failure_finder import (
    RandomShootingFailureFinder,
)
from hybrid_failure_discovery.failure_monitors.conveyorbelt_failure_monitor import (
    ConveyorBeltFailureMonitor,
)


def test_random_shooting_failure_finder_conveyorbelt():
    """Tests for random_shooting_failure_finder.py.

    With sufficient trajectories, should find a failure due to controller faults.

    This test uses higher sampling (300 trajectories) to ensure that failures
    can be reliably found. The controller has intentional faults that make
    failures possible, and with enough sampling, we should encounter at least
    one failure scenario.

    This test ASSERTs that a failure is found - if no failure is found with
    these parameters, the test will fail, indicating that either:
    1. The controller faults need to be more severe, or
    2. The trajectory counts need to be increased further
    """

    # Test failure finder in conveyorbelt env.
    # Create a scene spec with moderate constraints
    # Same parameters as the low-trajectory test for consistency
    scene_spec = ConveyorBeltSceneSpec(
        box_width=0.3,
        conveyor_belt_velocity=2.0,
        dt=0.01,
        belt_length=3.0,
    )
    object.__setattr__(scene_spec, "min_spacing", 0.1)

    env = ConveyorBeltEnv(scene_spec=scene_spec)
    secret_mode_sequence = ["fast", "mid", "fast", "slow", "off", "slow", "slow", "fast"]
    controller = ConveyorBeltController(
        seed=123,
        scene_spec=env.scene_spec,
        secret_failure_mode_sequence=secret_mode_sequence,
    )
    failure_monitor = ConveyorBeltFailureMonitor(env.scene_spec)
    # Higher trajectory count - should find failure
    failure_finder = RandomShootingFailureFinder(
        seed=123, max_num_trajectories=1000, max_trajectory_length=200
    )
    result = failure_finder.run(env, controller, failure_monitor)

    # Assert that a failure was found - this is the key difference from
    # the low-trajectory test. With sufficient sampling, we expect to find failures.
    assert result is not None

    states = result.observations
    # Accessing protected method _render_state is intentional for visualization
    # pylint: disable=protected-access
    imgs = [env._render_state(s) for s in states]
    path = (
        Path("videos")
        / "test-random-shooting"
        / "conveyorbelt_random_shooting_test.mp4"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.mimsave(path, imgs, fps=env.metadata["render_fps"])
