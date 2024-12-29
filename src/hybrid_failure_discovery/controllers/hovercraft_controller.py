"""A controller for the hovercraft environment."""

from tomsutils.gym_agent import Agent
from hybrid_failure_discovery.envs.hovercraft_env import HoverCraftState, HoverCraftAction, HoverCraftSceneSpec


class HoverCraftController(Agent[HoverCraftState, HoverCraftAction]):
    """A controller for the hovercraft environment."""

    def __init__(self, scene_spec: HoverCraftSceneSpec, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._scene_spec = scene_spec

    def _get_action(self) -> HoverCraftAction:
        import ipdb; ipdb.set_trace()

