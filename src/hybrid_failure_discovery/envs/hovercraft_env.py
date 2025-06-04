"""Hovercraft environment from Apurva Badithela."""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from gymnasium.core import RenderFrame
from numpy.typing import NDArray
from tomsgeoms2d.structs import Circle, Geom2D, Rectangle
from tomsutils.spaces import EnumSpace, FunctionalSpace
from tomsutils.utils import fig2data

from hybrid_failure_discovery.envs.constraint_based_env_model import (
    ConstraintBasedGymEnv,
)


@dataclass(frozen=True)
class HoverCraftState:
    """A state in the hovercraft environment."""

    x: float  # x position
    vx: float  # x velocity
    y: float  # y position
    vy: float  # y velocity
    t: float  # the current time in seconds


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
    init_x_range: tuple[float, float] = (-0.2, 0.2)
    init_vx: float = 0
    init_y_range: tuple[float, float] = (-0.2, 0.2)
    init_vy: float = 0

    init_goal_index: tuple[int, int] = (0, 0)  # index into goal_pairs

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

    # Goal pairs.
    goal_pairs: list[tuple[tuple[float, float], tuple[float, float]]] = field(
        default_factory=lambda: [
            ((-0.42, 0.0), (0.42, 0.0)),  # left, right
            (
                (
                    0.0,
                    -0.42,
                ),
                (0.0, 0.42),
            ),  # down, up
        ]
    )
    goal_switch_interval: float = 2.0  # goal allowed to switch after # seconds
    goal_atol: float = 1e-3

    # Rendering hyperparameters.
    render_figscale: float = 5
    render_padding: float = 0.05
    hovercraft_color: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    obstacle_color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)
    goal_circle_color: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.25)
    goal_star_color: tuple[float, float, float, float] = (1.0, 1.0, 0.8, 1.0)
    goal_star_size: float = 360

    @property
    def A(self) -> NDArray:
        """System dynamics A matrix."""
        return np.array(
            [
                [1, self.dt, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, self.dt],
                [0, 0, 0, 1],
            ]
        )

    @property
    def B(self) -> NDArray:
        """System dynamics B matrix."""
        return np.array(
            [
                [self.dt**2 / 2, 0],
                [self.dt, 0],
                [0, self.dt**2 / 2],
                [0, self.dt],
            ]
        )

    @property
    def Q(self) -> NDArray:
        """Cost function Q matrix."""
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.1, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.1],
            ]
        )

    @property
    def R(self) -> NDArray:
        """Cost function R matrix."""
        return 1e-2 * np.eye(2)


class HoverCraftEnv(ConstraintBasedGymEnv[HoverCraftState, HoverCraftAction]):
    """A 2D hovercraft environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        scene_spec: HoverCraftSceneSpec = HoverCraftSceneSpec(),
        seed: int = 0,
    ) -> None:
        self.scene_spec = scene_spec
        self.render_mode = "rgb_array"
        super().__init__(seed)
        # Create one RNG with the seed here
        self._rng = np.random.default_rng(seed)

    def _create_action_space(self) -> FunctionalSpace[HoverCraftAction]:
        return FunctionalSpace(
            contains_fn=lambda x: isinstance(x, HoverCraftAction),
        )

    def _get_obs(self) -> HoverCraftState:
        assert self._current_state is not None
        return self._current_state

    def get_initial_states(self) -> EnumSpace[HoverCraftState]:
        # Fully constrained for now.

        init_x = self._rng.uniform(*self.scene_spec.init_x_range)
        init_y = self._rng.uniform(*self.scene_spec.init_y_range)
        initial_state = HoverCraftState(
            x=init_x,
            vx=self.scene_spec.init_vx,
            y=init_y,
            vy=self.scene_spec.init_vy,
            t=0.0,
        )
        return EnumSpace([initial_state])

    def get_next_states(
        self, state: HoverCraftState, action: HoverCraftAction
    ) -> EnumSpace[HoverCraftState]:

        assert self.action_space.contains(action)

        A = self.scene_spec.A
        B = self.scene_spec.B

        # Integrate.
        state_vec = np.array([state.x, state.vx, state.y, state.vy])
        action_vec = np.array([action.ux, action.uy])
        next_state_vec = A @ state_vec + B @ action_vec
        x, vx, y, vy = next_state_vec

        # Advance time.
        dt = self.scene_spec.dt
        t = state.t + dt

        # Fully constrained.
        next_state = HoverCraftState(x=x, y=y, vx=vx, vy=vy, t=t)

        return EnumSpace([next_state])

    def actions_are_equal(
        self, action1: HoverCraftAction, action2: HoverCraftAction
    ) -> bool:
        return np.allclose([action1.ux, action1.uy], [action2.ux, action2.uy])

    def get_hovercraft_circle(self, state: HoverCraftState) -> Circle:
        """For rendering and external failure checking."""
        return Circle(state.x, state.y, self.scene_spec.hovercraft_radius)

    def _get_reward_and_termination(
        self,
        state: HoverCraftState,
        action: HoverCraftAction,
        next_state: HoverCraftState,
    ) -> tuple[float, bool]:
        return 0.0, False

    def _render_state(
        self, state: HoverCraftState
    ) -> RenderFrame | list[RenderFrame] | None:
        pad = self.scene_spec.render_padding
        min_x = -self.scene_spec.scene_width / 2 - pad
        max_x = self.scene_spec.scene_width / 2 + pad
        min_y = -self.scene_spec.scene_height / 2 - pad
        max_y = self.scene_spec.scene_height / 2 + pad

        scale = self.scene_spec.render_figscale
        fig, ax = plt.subplots(
            1, 1, figsize=(scale * (max_x - min_x), scale * (max_y - min_y))
        )

        # Plot hovercraft.
        circ = self.get_hovercraft_circle(state)
        circ.plot(ax, facecolor=self.scene_spec.hovercraft_color, edgecolor="black")

        # Plot obstacles.
        for obstacle in self.scene_spec.obstacles:
            obstacle.plot(
                ax, facecolor=self.scene_spec.obstacle_color, edgecolor="black"
            )

        # Plot all goals.
        for goal_pair in self.scene_spec.goal_pairs:
            for gx, gy in goal_pair:
                circ = Circle(gx, gy, self.scene_spec.hovercraft_radius)
                circ.plot(ax, facecolor=self.scene_spec.goal_circle_color)

        ax.set_xlim(min_x + pad, max_x - pad)
        ax.set_ylim(min_y + pad, max_y - pad)

        plt.tight_layout()

        img = fig2data(fig)
        plt.close()

        return img  # type: ignore
