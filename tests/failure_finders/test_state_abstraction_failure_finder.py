"""Tests for state_abstraction_failure_finder.py."""

from hybrid_failure_discovery.controllers.blocks_controller import (
    BlocksController,
)
from hybrid_failure_discovery.envs.blocks_env import (
    BlocksEnv,
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

    def blocks_abstract_heuristic(abstract_state_sequence):
        """Encourage building as tall a tower as possible."""
        max_tower_height = 0

        def _get_tower_height(block, support_graph):
            if block not in support_graph:
                return 1
            return 1 + max(
                _get_tower_height(above, support_graph)
                for above in support_graph[block]
            )

        for abstract_state in abstract_state_sequence:
            on_relations, _ = abstract_state

            # Build a graph representation of the "on" relationships
            support_graph = {}
            for top, bottom in on_relations:
                if bottom not in support_graph:
                    support_graph[bottom] = []
                support_graph[bottom].append(top)

            # Find the tallest tower

            # Compute the height for all base blocks
            for base_block in support_graph:
                max_tower_height = max(
                    max_tower_height, _get_tower_height(base_block, support_graph)
                )

        return 10 - max_tower_height

    controller = BlocksController(123, env.scene_spec, safe_height=0.2)
    failure_monitor = BlocksFailureMonitor()
    failure_finder = StateAbstractionFailureFinder(
        get_blocks_state_abstraction,
        blocks_abstract_heuristic,
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
    # path = (
    #     Path("videos")
    #     / "test-state-abstraction-failure-finding"
    #     / "state_abstraction_test.mp4"
    # )
    # path.parent.mkdir(exist_ok=True)
    # iio.mimsave(path, imgs, fps=env.metadata["render_fps"])
