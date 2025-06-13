"""Tests for annealing failure finder."""

import numpy as np

from hybrid_failure_discovery.controllers.hovercraft_controller import (
    HoverCraftController,
)
from hybrid_failure_discovery.envs.hovercraft_env import HoverCraftEnv
from hybrid_failure_discovery.failure_finders.blackbox_optimizers import (
    AnnealingFailureFinder,
)
from hybrid_failure_discovery.failure_monitors.hovercraft_failure_monitor import (
    HoverCraftFailureMonitor,
)


def test_annealing_failure_finder():
    """Tests for annealing_failure_finder.py."""

    # Test failure finder in hovercraft env.
    env = HoverCraftEnv()
    controller = HoverCraftController(123, env.scene_spec)
    failure_monitor = HoverCraftFailureMonitor(env.scene_spec)
    # failure_finder = AnnealingFailureFinder(seed=123)
    # result = failure_finder.run(env, controller, failure_monitor)

    annealer = AnnealingFailureFinder(
        env=env, controller=controller, failure_monitor=failure_monitor, seed=123
    )
    annealer.Tmax = 1.0  # Lower starting temperature
    annealer.Tmin = 0.001  # Stop sooner
    annealer.steps = 10  # Reduce number of iterations
    best_state, best_energy = annealer.anneal()

    # Test failure finder in blocks env with a too-low height.
    # env = BlocksEnv()
    # controller = BlocksController(123, env.scene_spec, safe_height=0.15)
    # failure_monitor = BlocksFailureMonitor()
    # failure_finder = AnnealingFailureFinder(seed=123, max_trajectory_length=250)
    # result = failure_finder.run(env, controller, failure_monitor)
    # assert result is not None

    # Uncomment to visualize.
    from pathlib import Path

    import imageio.v2 as iio

    states = annealer.hc_state_traj  # Hovercraft trajectory
    imgs = [env._render_state(s) for s in states]

    # Find the root directory of the repository (the directory containing the .git folder)
    repo_root = Path(__file__).resolve().parent
    while not (repo_root / ".git").exists():  # Look for the .git folder
        repo_root = repo_root.parent

    # Now construct the path relative to the repo root
    path = repo_root / "videos" / "test-annealing-shooting" / "annealing_test.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.mimsave(path, imgs, fps=env.metadata["render_fps"])
