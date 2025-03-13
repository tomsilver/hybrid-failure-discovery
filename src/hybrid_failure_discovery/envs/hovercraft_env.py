"""Hovercraft environment from Apurva Badithela."""

from dataclasses import dataclass, field
from pdb import set_trace as st

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
    gx: float  # goal x position
    gy: float  # goal y position
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

    def get_goal_pair_index_from_state(self, state: HoverCraftState) -> tuple[int, int]:
        """Get the goal pair index from a state."""
        obs_goal = np.array([state.gx, state.gy])
        for i, pair in enumerate(self.goal_pairs):
            for j, goal in enumerate(pair):
                if np.allclose(goal, obs_goal):
                    return (i, j)
        raise ValueError(f"Unrecognized goal: {obs_goal}")


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

    def _create_action_space(self) -> FunctionalSpace[HoverCraftAction]:
        return FunctionalSpace(
            contains_fn=lambda x: isinstance(x, HoverCraftAction),
        )

    def _get_obs(self) -> HoverCraftState:
        assert self._current_state is not None
        return self._current_state

    def get_initial_states(self) -> EnumSpace[HoverCraftState]:
        # Fully constrained for now.
        gi, gj = self.scene_spec.init_goal_index
        gx, gy = self.scene_spec.goal_pairs[gi][gj]
        initial_state = HoverCraftState(
            x=self.scene_spec.init_x,
            vx=self.scene_spec.init_vx,
            y=self.scene_spec.init_y,
            vy=self.scene_spec.init_vy,
            gx=gx,
            gy=gy,
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

        # Handle the goal, which is the only part that is not fully constrained.
        gi, gj = self.scene_spec.get_goal_pair_index_from_state(state)
        gx, gy = self.scene_spec.goal_pairs[gi][gj]
        goal_vec = np.array([gx, 0, gy, 0])

        # First, always toggle if reached goal.
        if np.allclose(goal_vec, state_vec, atol=self.scene_spec.goal_atol):
            gj = int(not gj)
            gx, gy = self.scene_spec.goal_pairs[gi][gj]

        # The next state might have the same goal.
        next_states = [
            HoverCraftState(x=x, y=y, vx=vx, vy=vy, gx=gx, gy=gy, t=t),
        ]

        # Or the next state might have a switched goal.
        switch_intv = self.scene_spec.goal_switch_interval
        if np.floor(t / switch_intv) > np.floor((t - dt) / switch_intv):
            switch_gi = int(not gi)
            switch_gj = 0  # arbitrarily always start with left or down
            switch_gx, switch_gy = self.scene_spec.goal_pairs[switch_gi][switch_gj]
            switch_next_state = HoverCraftState(
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                gx=switch_gx,
                gy=switch_gy,
                t=t,
            )
            next_states.append(switch_next_state)

        return EnumSpace(next_states)

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
        gx = next_state.gx
        gy = next_state.gy
        goal_vec = np.array([gx, 0, gy, 0])
        next_state_vec = np.array(
            [next_state.x, next_state.vx, next_state.y, next_state.vy]
        )
        action_vec = np.array([action.ux, action.uy])
        error_vec = np.subtract(next_state_vec, goal_vec)
        Q = self.scene_spec.Q
        R = self.scene_spec.R
        cost = error_vec.T @ Q @ error_vec + action_vec.T @ R @ action_vec
        reward = -cost
        return reward, False

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
