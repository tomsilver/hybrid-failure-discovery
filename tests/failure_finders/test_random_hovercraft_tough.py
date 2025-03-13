"""Tests for random_shooting_failure_finder.py."""

from pdb import set_trace as st

from tomsgeoms2d.structs import geom2ds_intersect

from hybrid_failure_discovery.controllers.hovercraft_controller_tough import (
    HoverCraftControllerTough,
)
from hybrid_failure_discovery.envs.hovercraft_env_tough import (
    HoverCraftEnvTough,
    HoverCraftState,
)
from hybrid_failure_discovery.failure_finders.failure_finder import (
    MemorylessStateFailureMonitor,
)
from hybrid_failure_discovery.failure_finders.random_shooting_failure_finder import (
    RandomShootingFailureFinder,
)


def test_random_shooting_failure_finder_tough():
    """Tests for random_shooting_failure_finder.py."""

    env = HoverCraftEnvTough()
    controller = HoverCraftControllerTough(env.scene_spec)

    def _collision_failure(state: HoverCraftState) -> bool:
        circ = env.get_hovercraft_circle(state)
        for obstacle in env.scene_spec.obstacles:
            if geom2ds_intersect(circ, obstacle):
                return True
        return False

    failure_monitor = MemorylessStateFailureMonitor(_collision_failure)

    failure_finder = RandomShootingFailureFinder(seed=123)

    result = failure_finder.run(env, controller, failure_monitor)
    assert result is not None

    # Uncomment to visualize.
    from pathlib import Path

    import imageio.v2 as iio

    states, _ = result
    imgs = [env._render_state(s) for s in states]

    # Find the root directory of the repository (the directory containing the .git folder)
    repo_root = Path(__file__).resolve().parent
    while not (repo_root / ".git").exists():  # Look for the .git folder
        repo_root = repo_root.parent

    # Now construct the path relative to the repo root
    path = (
        repo_root
        / "videos"
        / "test-random-shooting"
        / "random-shooting_test_hovercraft_tough.mp4"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.mimsave(path, imgs, fps=env.metadata["render_fps"])
