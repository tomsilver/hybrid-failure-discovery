"""Tests for oracle_commander_failure_finder.py."""

from hybrid_failure_discovery.commander.commander import Commander
from hybrid_failure_discovery.commander.initial_state_commander import (
    InitialStateCommander,
)
from hybrid_failure_discovery.controllers.hovercraft_controller import (
    HoverCraftCommand,
    HoverCraftController,
)
from hybrid_failure_discovery.envs.hovercraft_env import (
    HoverCraftEnv,
    HoverCraftSceneSpec,
    HoverCraftState,
)
from hybrid_failure_discovery.failure_finders.oracle_commander_failure_finder import (
    OracleCommanderFailureFinder,
)
from hybrid_failure_discovery.failure_monitors.hovercraft_failure_monitor import (
    HoverCraftFailureMonitor,
)


def test_oracle_initial_state_and_commander_failure_finder() -> None:
    """Tests for oracle_commander_failure_finder.py."""

    # Test failure finder in hovercraft env.

    class _OracleHoverCraftCommander(Commander):
        """An oracle commander for the hovercraft environment."""

        def __init__(self, scene_spec: HoverCraftSceneSpec, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._scene_spec = scene_spec
            self._current_state = None
            self._switched = False

        def reset(self, initial_state):
            self._current_state = initial_state

        def get_command(self):
            # Force a switch when near the right goal.
            assert isinstance(self._current_state, HoverCraftState)
            current_x = self._current_state.x
            goal_x = self._scene_spec.goal_pairs[0][1][0]
            dist = abs(current_x - goal_x)
            if dist < 0.1 and not self._switched:
                self._switched = True
                return HoverCraftCommand(switch=True)
            return HoverCraftCommand(switch=False)

        def update(self, action, next_state):
            self._current_state = next_state

    class _OracleHoverCraftInitialStateCommander(InitialStateCommander):
        """An oracle commander for the hovercraft environment."""

        def __init__(self, scene_spec: HoverCraftSceneSpec, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._scene_spec = scene_spec

        def initialize(self):
            # Hard-coded initial state for testing
            init_x, init_y = self._scene_spec.goal_pairs[0][0]
            initial_state = HoverCraftState(
                x=init_x,
                vx=self._scene_spec.init_vx,
                y=init_y,
                vy=self._scene_spec.init_vy,
                t=0.0,
            )
            breakpoint()  # Set a breakpoint here to inspect the initial state.
            return initial_state

    env = HoverCraftEnv()
    controller = HoverCraftController(123, env.scene_spec)
    failure_monitor = HoverCraftFailureMonitor(env.scene_spec)
    initial_state_commander = _OracleHoverCraftInitialStateCommander(env.scene_spec)
    commander = _OracleHoverCraftCommander(env.scene_spec)
    failure_finder = OracleCommanderFailureFinder(commander, seed=123)
    result = failure_finder.run(env, controller, failure_monitor)
    assert result is not None

    # Uncomment to visualize.
    # from pathlib import Path
    # import imageio.v2 as iio
    # states = result.observations
    # imgs = [env._render_state(s) for s in states]
    # path = Path("videos") / "test-oracle-commander" / "oracle_commander_test.mp4"
    # path.parent.mkdir(exist_ok=True)
    # iio.mimsave(path, imgs, fps=env.metadata["render_fps"])
