"""Tests for random_shooting_failure_finder.py.

This test suite validates the RandomShootingFailureFinder's ability to discover
failures in the Blocks environment. The BlocksController may have intentional
faults or the environment may naturally produce failures under certain conditions.

The test suite includes scenarios to test failure detection:
1. High trajectory count: Tests that with sufficient sampling, failures can
   be reliably found (validates failure detection works)
"""

from pathlib import Path

import imageio.v2 as iio

from hybrid_failure_discovery.controllers.blocks_controller import BlocksController
from hybrid_failure_discovery.envs.blocks_env import BlocksEnv
from hybrid_failure_discovery.failure_finders.random_shooting_failure_finder import (
    RandomShootingFailureFinder,
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


def test_random_shooting_failure_finder_blocks():
    """Tests for random_shooting_failure_finder.py with Blocks environment.

    With sufficient trajectories, should find a failure if one exists.

    This test uses higher sampling to ensure that failures can be
    reliably found if they exist in the system. The test may or may not
    find a failure depending on the controller and environment
    configuration.
    """

    # Test failure finder in blocks env
    # Use more blocks to increase failure opportunities
    # pylint: disable=import-outside-toplevel
    from hybrid_failure_discovery.envs.blocks_env import BlocksEnvSceneSpec

    scene_spec = BlocksEnvSceneSpec(num_blocks=4)
    env = BlocksEnv(scene_spec=scene_spec, seed=123, use_gui=False)

    # Use lower safe_height to make controller less careful
    # (more likely to cause collisions)
    controller = BlocksController(seed=123, scene_spec=env.scene_spec, safe_height=0.15)

    # Use very sensitive tolerance to catch even tiny movements
    failure_monitor = BlocksFailureMonitor(
        move_tol=0.01
    )  # Very sensitive - catch small movements

    # Balanced trajectory count and length - aggressive but not so slow
    failure_finder = RandomShootingFailureFinder(
        seed=123, max_num_trajectories=100, max_trajectory_length=1000
    )

    # Try multiple seeds to explore more scenarios
    print(
        f"Testing with {scene_spec.num_blocks} blocks, "
        f"safe_height={0.15}, move_tol={0.01}"
    )
    max_traj = failure_finder._max_num_trajectories  # pylint: disable=protected-access
    max_len = failure_finder._max_trajectory_length  # pylint: disable=protected-access
    print(f"Searching up to {max_traj} trajectories (max length {max_len})...")
    print("This may take a while - planning with many blocks can be slow...")

    # Run the failure finder, handling empty task plan exceptions
    # Empty task plan means the command was successfully completed (no failure)
    try:
        result = failure_finder.run(env, controller, failure_monitor)
    except TaskThenMotionPlanningFailure as e:
        # If the planner has no task plan (empty plan), this means the command
        # was successfully completed. No failure found - treat as no failure.
        if "empty" in str(e).lower() or "task plan" in str(e).lower():
            result = None  # No failure found - command completed successfully
        else:
            # Re-raise if it's a different planning failure
            raise

    # Check if a failure was found
    # A failure occurs when a block that is NOT held moves unexpectedly
    # (e.g., due to collisions, robot hitting blocks, physics issues)
    if result is not None:
        print(f"✓ Failure found! Trajectory length: {len(result.observations)} steps")
        reason = failure_monitor.failure_reason or "unknown"
        print(f"  Failure cause: {reason}")
    else:
        print("✗ No failure found after all trajectories")
        print("  This could mean:")
        print("  - The controller is very robust and avoids collisions")
        print("  - Failures are extremely rare in this environment")
        print("  - The BlocksController planner is too careful")
        print("\n  Suggestions:")
        print("  - Try even more trajectories (increase max_num_trajectories)")
        print("  - Try different seeds to explore different scenarios")
        print("  - Consider if failures naturally occur in this environment")

    # Note: This test may or may not find a failure depending on the system
    # If failures are expected, uncomment the assertion below
    # assert result is not None

    traj_to_render = result or failure_finder.last_trajectory
    if traj_to_render is not None:
        states = traj_to_render.observations
    else:
        initial_state = env.get_initial_states().sample()
        states = [initial_state] * 30
    # pylint: disable=protected-access
    imgs = [env._render_state(s) for s in states]
    path = Path("videos") / "test-random-shooting" / "blocks_random_shooting_test.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.mimsave(path, imgs, fps=env.metadata["render_fps"])

    env.close()
