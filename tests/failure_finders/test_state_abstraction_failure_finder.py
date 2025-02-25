"""Tests for state_abstraction_failure_finder.py."""

from hybrid_failure_discovery.controllers.blocks_controller import (
    BlocksController,
)
from hybrid_failure_discovery.envs.blocks_env import (
    BlocksEnv,
    BlocksEnvSceneSpec,
    BlocksEnvState,
)
from hybrid_failure_discovery.failure_finders.state_abstraction_failure_finder import (
    StateAbstractionFailureFinder,
)
from hybrid_failure_discovery.failure_monitors.blocks_failure_monitor import (
    BlocksFailureMonitor,
)


def test_state_abstraction_failure_finder():
    """Tests for state_abstraction_failure_finder.py."""

    # Test failure finder in blocks env.

    env = BlocksEnv(use_gui=False)

    def get_blocks_state_abstraction(traj):
        """What blocks are on what blocks, and what is held."""
        thresh = 1e-3
        height = env.scene_spec.block_half_extents[2]
        state = traj[0][-1]
        assert isinstance(state, BlocksEnvState)
        held_block = state.held_block_name
        on_relations = set()
        for b1 in state.blocks:
            x1, y1, z1 = b1.pose.position
            for b2 in state.blocks:
                x2, y2, z2 = b2.pose.position
                if (
                    abs(x1 - x2) < thresh
                    and abs(y1 - y2) < thresh
                    and abs(z1 + 2 * height - z2) < thresh
                ):
                    on_relations.add((b1.name, b2.name))
        return frozenset(on_relations), held_block

    controller = BlocksController(123, env.scene_spec, safe_height=0.2)
    failure_monitor = BlocksFailureMonitor()
    failure_finder = StateAbstractionFailureFinder(
        get_blocks_state_abstraction,
        seed=123,
        max_trajectory_length=1000,
    )
    result = failure_finder.run(env, controller, failure_monitor)
    assert result is not None

    # Uncomment to visualize.
    from pathlib import Path
    import imageio.v2 as iio
    states, _ = result
    imgs = [env._render_state(s) for s in states]
    path = (
        Path("videos")
        / "test-state-abstraction-failure-finding"
        / "state_abstraction_test.mp4"
    )
    path.parent.mkdir(exist_ok=True)
    iio.mimsave(path, imgs, fps=env.metadata["render_fps"])
