"""Tests for heuristic_failure_finder.py."""

from hybrid_failure_discovery.controllers.hovercraft_controller import (
    HoverCraftController,
)
from hybrid_failure_discovery.envs.hovercraft_env import HoverCraftEnv
from hybrid_failure_discovery.failure_finders.heuristic_failure_finder import (
    HeuristicFailureFinder,
)
from hybrid_failure_discovery.failure_monitors.hovercraft_failure_monitor import (
    HoverCraftFailureMonitor,
)


def test_heuristic_failure_finder():
    """Tests for heuristic_failure_finder.py."""

    # Test failure finder in hovercraft env.

    def hovercraft_heuristic(traj):
        """Encourage two switches back-to-back."""
        states, actions = traj
        switch_just_happened = False
        for t in range(len(actions)):
            state = states[t]
            next_state = states[t + 1]
            switch_happening = state.gx != next_state.gx or state.gy != next_state.gy
            if switch_happening and switch_just_happened:
                return 0.0  # good!
            switch_just_happened = switch_happening
        return 1.0  # bad

    env = HoverCraftEnv()
    controller = HoverCraftController(123, env.scene_spec)
    failure_monitor = HoverCraftFailureMonitor(env.scene_spec)
    failure_finder = HeuristicFailureFinder(hovercraft_heuristic, seed=123)
    result = failure_finder.run(env, controller, failure_monitor)
    assert result is not None

    # Uncomment to visualize.
    # from pathlib import Path
    # import imageio.v2 as iio
    # states = result.observations
    # imgs = [env._render_state(s) for s in states]
    # path = Path("videos") / "test-heuristic-failure-finding" / "heuristic_test.mp4"
    # path.parent.mkdir(exist_ok=True)
    # iio.mimsave(path, imgs, fps=env.metadata["render_fps"])
