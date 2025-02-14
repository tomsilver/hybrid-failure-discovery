"""Tests for random_shooting_failure_finder.py."""

from hybrid_failure_discovery.controllers.blocks_controller import (
    BlocksController,
)
from hybrid_failure_discovery.controllers.hovercraft_controller import (
    HoverCraftController,
)
from hybrid_failure_discovery.envs.blocks_env import (
    BlocksEnv,
)
from hybrid_failure_discovery.envs.hovercraft_env import HoverCraftEnv
from hybrid_failure_discovery.failure_finders.random_shooting_failure_finder import (
    RandomShootingFailureFinder,
)
from hybrid_failure_discovery.failure_monitors.blocks_failure_monitor import (
    BlocksFailureMonitor,
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

    # Test failure finder in blocks env with a too-low height.
    env = BlocksEnv()
    controller = BlocksController(123, env.scene_spec, safe_height=0.15)
    failure_monitor = BlocksFailureMonitor()
    failure_finder = RandomShootingFailureFinder(seed=123, max_trajectory_length=250)
    result = failure_finder.run(env, controller, failure_monitor)
    assert result is not None

    # Uncomment to visualize.
    # from pathlib import Path
    # import imageio.v2 as iio
    # states, _ = result
    # imgs = [env._render_state(s) for s in states]
    # path = Path("videos") / "test-random-shooting" / "random-shooting_test.mp4"
    # path.parent.mkdir(exist_ok=True)
    # iio.mimsave(path, imgs, fps=env.metadata["render_fps"])
