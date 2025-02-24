"""Tests for heuristic_failure_finder.py."""

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
from hybrid_failure_discovery.failure_finders.heuristic_failure_finder import (
    HeuristicFailureFinder,
)
from hybrid_failure_discovery.failure_monitors.blocks_failure_monitor import (
    BlocksFailureMonitor,
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

    # Test failure finder in blocks env with a too-low height.

    def blocks_heuristic(traj):
        """Encourage building as tall a tower as possible."""
        states, _ = traj
        tallest_tower_height = 0.0
        for state in states:
            # Don't count states where a block is held.
            if state.held_block_name is not None:
                continue
            # Find the highest block.
            for block_state in state.blocks:
                z = block_state.pose.position[2]
                tallest_tower_height = max(tallest_tower_height, z)
        assert tallest_tower_height <= 1.0
        return 1.0 - tallest_tower_height  # lower is better

    env = BlocksEnv(use_gui=False)
    controller = BlocksController(123, env.scene_spec, safe_height=0.2)
    failure_monitor = BlocksFailureMonitor()
    failure_finder = HeuristicFailureFinder(
        blocks_heuristic,
        seed=123,
        max_trajectory_length=1000,
    )
    result = failure_finder.run(env, controller, failure_monitor)
    assert result is not None

    # Uncomment to visualize.
    # from pathlib import Path
    # import imageio.v2 as iio
    # states, _ = result
    # imgs = [env._render_state(s) for s in states]
    # path = Path("videos") / "test-heuristic-failure-finding" / "heuristic_test.mp4"
    # path.parent.mkdir(exist_ok=True)
    # iio.mimsave(path, imgs, fps=env.metadata["render_fps"])
