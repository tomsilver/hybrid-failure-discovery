"""Hovercraft environment from Apurva Badithela."""

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from gymnasium.core import Env, RenderFrame
from numpy.typing import NDArray
from tomsgeoms2d.structs import Circle, Geom2D, Rectangle
from tomsutils.spaces import FunctionalSpace
from tomsutils.utils import fig2data


@dataclass(frozen=True)
class HoverCraftState:
    """A state in the hovercraft environment."""

    x: float  # x position
    vx: float  # x velocity
    y: float  # y position
    vy: float  # y velocity
    gx: float  # goal x position
    gy: float  # goal y position


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
    init_vx: float = 0
    init_y: float = 0
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
    goal_pair_switch_prob: float = 0.01
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

    def get_goal_pair_index_from_state(self, state: HoverCraftState) -> tuple[int, int]:
        """Get the goal pair index from a state."""
        obs_goal = np.array([state.gx, state.gy])
        for i, pair in enumerate(self.goal_pairs):
            for j, goal in enumerate(pair):
                if np.allclose(goal, obs_goal):
                    return (i, j)
        raise ValueError(f"Unrecognized goal: {obs_goal}")


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
        gi, gj = self.scene_spec.init_goal_index
        gx, gy = self.scene_spec.goal_pairs[gi][gj]
        self._current_state = HoverCraftState(
            x=self.scene_spec.init_x,
            vx=self.scene_spec.init_vx,
            y=self.scene_spec.init_y,
            vy=self.scene_spec.init_vy,
            gx=gx,
            gy=gy,
        )

        return self._get_state(), self._get_info()

    def step(
        self, action: HoverCraftAction
    ) -> tuple[HoverCraftState, float, bool, bool, dict[str, Any]]:

        assert self.action_space.contains(action)

        state = self._get_state()

        A = self.scene_spec.A
        B = self.scene_spec.B
        Q = self.scene_spec.Q
        R = self.scene_spec.R

        # Integrate.
        state_vec = np.array([state.x, state.vx, state.y, state.vy])
        action_vec = np.array([action.ux, action.uy])
        next_state_vec = A @ state_vec + B @ action_vec
        x, vx, y, vy = next_state_vec

        # Calculate reward (inverse cost).
        gx = state.gx
        gy = state.gy
        goal_vec = np.array([gx, 0, gy, 0])
        error_vec = np.subtract(state_vec, goal_vec)
        cost = error_vec.T @ Q @ error_vec + action_vec.T @ R @ action_vec
        reward = -cost

        # Handle the goal.
        gi, gj = self.scene_spec.get_goal_pair_index_from_state(state)

        # Randomly reset between up/down and left/right goals.
        if self._rng.uniform() < self.scene_spec.goal_pair_switch_prob:
            gi_choices = [i for i in range(len(self.scene_spec.goal_pairs)) if i != gi]
            gi = self._rng.choice(gi_choices)
            gj = self._rng.choice(2)
            gx, gy = self.scene_spec.goal_pairs[gi][gj]

        # Toggle the goal if we've reached it.
        elif np.allclose(goal_vec, state_vec, atol=self.scene_spec.goal_atol):
            # Switch to other goal in the pair.
            gj = int(not gj)
            gx, gy = self.scene_spec.goal_pairs[gi][gj]

        self._current_state = HoverCraftState(x=x, y=y, vx=vx, vy=vy, gx=gx, gy=gy)

        return self._get_state(), reward, False, False, self._get_info()

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

        # Plot all goals.
        for goal_pair in self.scene_spec.goal_pairs:
            for gx, gy in goal_pair:
                circ = Circle(gx, gy, self.scene_spec.hovercraft_radius)
                circ.plot(ax, facecolor=self.scene_spec.goal_circle_color)

        # Plot the current goal.
        ax.scatter(
            [state.gx],
            [state.gy],
            s=self.scene_spec.goal_star_size,
            marker="*",
            color=self.scene_spec.goal_star_color,
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
