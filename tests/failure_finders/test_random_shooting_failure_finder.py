"""Tests for random_shooting_failure_finder.py."""

# Uncomment to visualize.
from pathlib import Path

import imageio.v2 as iio

from hybrid_failure_discovery.controllers.hovercraft_controller import (
    HoverCraftController,
)
from hybrid_failure_discovery.envs.hovercraft_env import HoverCraftEnv
from hybrid_failure_discovery.failure_finders.random_shooting_failure_finder import (
    RandomShootingFailureFinder,
)
from hybrid_failure_discovery.failure_monitors.hovercraft_failure_monitor import (
    HoverCraftFailureMonitor,
)


def test_random_shooting_failure_finder():
    """Tests for random_shooting_failure_finder.py."""

    # Test failure finder in hovercraft env.
    env = HoverCraftEnv()
    controller = HoverCraftController(123, env.scene_spec)
    failure_monitor = HoverCraftFailureMonitor(env.scene_spec)
    failure_finder = RandomShootingFailureFinder(seed=123)
    result = failure_finder.run(env, controller, failure_monitor)
    assert result is not None

    states = result.observations
    # Accessing protected method _render_state is intentional for visualization
    # pylint: disable=protected-access
    imgs = [env._render_state(s) for s in states]
    path = Path("videos") / "test-random-shooting" / "random-shooting_test.mp4"
    path.parent.mkdir(exist_ok=True)
    iio.mimsave(path, imgs, fps=env.metadata["render_fps"])
