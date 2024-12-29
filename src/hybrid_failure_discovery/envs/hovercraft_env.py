"""Hovercraft environment from Apurva Badithela."""

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from gymnasium.core import Env, RenderFrame
from tomsgeoms2d.structs import Circle, Geom2D, Rectangle
from tomsutils.spaces import FunctionalSpace
from tomsutils.utils import fig2data


@dataclass(frozen=True)
class HoverCraftState:
    """A state in the hovercraft environment."""

    x: float  # x position
    y: float  # y position
    vx: float  # x velocity
    vy: float  # y velocity


@dataclass(frozen=True)
class HoverCraftAction:
    """An action in the hovercraft environment."""

    ux: float  # x acceleration
    uy: float  # y acceleration


@dataclass(frozen=True)
class HoverCraftSceneSpec:
    """Static hyperparameters for the hovercraft environment."""

    # Simulation stepsize.
    dt: float = 0.1

    # Initial state hyperparameters.
    init_x: float = 0
    init_y: float = 0
    init_vx: float = 0
    init_vy: float = 0

    # Scene hyperparameters.
    scene_width: float = 1
    scene_height: float = 1

    # Hovercraft hyperparameters.
    hovercraft_radius: float = 0.04

    # Obstacle hyperparameters.
    obstacles: list[Geom2D] = field(
        default_factory=lambda: [
            Rectangle(-0.5, -0.5, 0.3, 0.3, 0.0),  # bottom left
            Rectangle(0.2, -0.5, 0.3, 0.3, 0.0),  # bottom right
            Rectangle(0.2, 0.2, 0.3, 0.3, 0.0),  # top right
            Rectangle(-0.5, 0.2, 0.3, 0.3, 0.0),  # top left
        ]
    )

    # Rendering hyperparameters.
    render_figscale: float = 5
    render_padding: float = 0.05
    hovercraft_color: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    obstacle_color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)


class HoverCraftEnv(Env[HoverCraftState, HoverCraftAction]):
    """A 2D hovercraft environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        scene_spec: HoverCraftSceneSpec = HoverCraftSceneSpec(),
        seed: int = 0,
    ) -> None:

        self.scene_spec = scene_spec
        self._rng = np.random.default_rng(seed)

        self.render_mode = "rgb_array"
        self.action_space = FunctionalSpace(
            contains_fn=lambda x: isinstance(x, HoverCraftAction),
        )

        self._current_state: HoverCraftState | None = None  # set in reset()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[HoverCraftState, dict[str, Any]]:

        super().reset(seed=seed, options=options)

        # No randomization for now.
        self._current_state = HoverCraftState(
            x=self.scene_spec.init_x,
            y=self.scene_spec.init_y,
            vx=self.scene_spec.init_vx,
            vy=self.scene_spec.init_vy,
        )

        return self._get_state(), self._get_info()

    def step(
        self, action: HoverCraftAction
    ) -> tuple[HoverCraftState, float, bool, bool, dict[str, Any]]:

        assert self.action_space.contains(action)
        dt = self.scene_spec.dt

        # Move hovercraft.
        state = self._get_state()

        vx = state.vx + dt * action.ux
        vy = state.vy + dt * action.uy
        x = state.x + dt * vx
        y = state.y + dt * vy

        self._current_state = HoverCraftState(x=x, y=y, vx=vx, vy=vy)

        return self._get_state(), 0.0, False, False, self._get_info()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pad = self.scene_spec.render_padding
        min_x = -self.scene_spec.scene_width / 2 - pad
        max_x = self.scene_spec.scene_width / 2 + pad
        min_y = -self.scene_spec.scene_height / 2 - pad
        max_y = self.scene_spec.scene_height / 2 + pad

        scale = self.scene_spec.render_figscale
        fig, ax = plt.subplots(
            1, 1, figsize=(scale * (max_x - min_x), scale * (max_y - min_y))
        )

        # Get current state.
        state = self._get_state()

        # Plot hovercraft.
        circ = Circle(state.x, state.y, self.scene_spec.hovercraft_radius)
        circ.plot(ax, facecolor=self.scene_spec.hovercraft_color, edgecolor="black")

        # Plot obstacles.
        for obstacle in self.scene_spec.obstacles:
            obstacle.plot(
                ax, facecolor=self.scene_spec.obstacle_color, edgecolor="black"
            )

        ax.set_xlim(min_x + pad, max_x - pad)
        ax.set_ylim(min_y + pad, max_y - pad)

        plt.tight_layout()

        img = fig2data(fig)
        plt.close()

        return img  # type: ignore

    def _get_state(self) -> HoverCraftState:
        assert self._current_state is not None
        return self._current_state

    def _get_info(self) -> dict[str, Any]:
        return {}
