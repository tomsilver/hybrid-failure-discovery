# """Tests for random_shooting_failure_finder.py (ConveyorBelt)."""

# from hybrid_failure_discovery.controllers.conveyorbelt_controller import (
#     ConveyorBeltController,
# )
# from hybrid_failure_discovery.envs.conveyorbelt_env import ConveyorBeltEnv
# from hybrid_failure_discovery.failure_finders.random_shooting_failure_finder import (
#     RandomShootingFailureFinder,
# )
# from hybrid_failure_discovery.failure_monitors.conveyorbelt_failure_monitor import (
#     ConveyorBeltFailureMonitor,
# )


# def test_random_shooting_failure_finder_conveyorbelt():
#     """Tests for random_shooting_failure_finder.py on the ConveyorBelt environment."""

#     # Set up ConveyorBelt env + controller + failure monitor
#     env = ConveyorBeltEnv()
#     controller = ConveyorBeltController(seed=123, scene_spec=env.scene_spec)
#     failure_monitor = ConveyorBeltFailureMonitor(env.scene_spec)

#     # Run the generic random shooting failure finder
#     failure_finder = RandomShootingFailureFinder(seed=123)
#     result = failure_finder.run(env, controller, failure_monitor)
#     assert result is not None

#     # Uncomment to visualize the discovered trajectory as a video
#     from pathlib import Path
#     import imageio.v2 as iio
#     states = result.observations
#     imgs = [env._render_state(s) for s in states]
#     path = Path("videos") / "test-random-shooting" / "conveyorbelt_random_shooting_test.mp4"
#     path.parent.mkdir(parents=True, exist_ok=True)
#     iio.mimsave(path, imgs, fps=env.metadata["render_fps"])
"""
Tests for random_shooting_failure_finder.py (ConveyorBelt) with the SAFE controller.
This test is NOT expected to always find a failure — the controller prevents
most collisions — so we only check that it runs correctly and returns *some*
trajectory object (failed or not).
"""

from hybrid_failure_discovery.controllers.conveyorbelt_controller import (
    ConveyorBeltController,
)
from hybrid_failure_discovery.envs.conveyorbelt_env import ConveyorBeltEnv
from hybrid_failure_discovery.failure_finders.random_shooting_failure_finder import (
    RandomShootingFailureFinder,
)
from hybrid_failure_discovery.failure_monitors.conveyorbelt_failure_monitor import (
    ConveyorBeltFailureMonitor,
)


def test_random_shooting_failure_finder_conveyorbelt():
    """Smoke test that random-shooting executes cleanly under the ConveyorBelt controller."""

    # Environment, controller, failure monitor
    env = ConveyorBeltEnv()
    controller = ConveyorBeltController(seed=123, scene_spec=env.scene_spec)
    failure_monitor = ConveyorBeltFailureMonitor(env.scene_spec)

    # More trajectories: failure is now RARE because the controller is safe
    failure_finder = RandomShootingFailureFinder(seed=123, num_trajectories=30)

    # This returns a FailureFinderResult — even if no failure is found
    result = failure_finder.run(env, controller, failure_monitor)

    # Ensure it returned a valid result object
    assert result is not None
    assert hasattr(result, "observations")

    # Print (rather than assert) the failure info
    print("Random-shooting result:")
    print("  failure_found =", result.failure_found)
    print("  trajectory length =", len(result.observations))

    # Uncomment to save the discovered trajectory as a video
    
    from pathlib import Path
    import imageio.v2 as iio
    states = result.observations
    imgs = [env._render_state(s) for s in states]
    path = Path("videos") / "test-random-shooting" / "conveyorbelt_random_shooting_test.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.mimsave(path, imgs, fps=env.metadata["render_fps"])
    
