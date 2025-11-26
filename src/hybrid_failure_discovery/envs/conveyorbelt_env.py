"""This module defines the ConveyorBeltEnv environment for simulating a 1D
conveyor belt with packages that drop onto the conveyer belt at controllable
times."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from gymnasium.core import RenderFrame
from numpy.typing import NDArray
from tomsutils.spaces import EnumSpace, FunctionalSpace
from tomsutils.utils import fig2data

from hybrid_failure_discovery.envs.constraint_based_env_model import (
    ConstraintBasedGymEnv,
)


@dataclass(frozen=True)
class ConveyorBeltState:
    """A state in the conveyer belt environment."""

    positions: NDArray[np.float32]  # positions of packages on the conveyer belt
    falling_heights: NDArray[np.float32]  # height above belt (0 if on belt)
    step_count: int  # number of steps elapsed, mainly for visualization


@dataclass(frozen=True)
class ConveyorBeltAction:
    """An action in the conveyer belt environment."""

    drop_package: bool  # whether to drop a new package at the current time step


@dataclass(frozen=True)
class ConveyorBeltSceneSpec:
    """A spec for the conveyer belt environment."""

    conveyor_belt_velocity: float = 1.0
    dt: float = 0.01
    belt_length: float = 5.0
    box_width: float = 0.4
    box_height: float = 0.5
    gravity: float = 9.81
    drop_start_height: float = 1.0
    initial_drop_position: float = 0.0


class ConveyorBeltEnv(ConstraintBasedGymEnv[ConveyorBeltState, ConveyorBeltAction]):
    """1D conveyor belt environment with external control for dropping
    packages."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str = "rgb_array",
        scene_spec: ConveyorBeltSceneSpec = ConveyorBeltSceneSpec(),
        seed: int = 0,
    ):
        self.scene_spec = scene_spec
        self.render_mode = render_mode

        self._rng = np.random.default_rng(seed)

        super().__init__(seed)

    def _create_action_space(self) -> FunctionalSpace[ConveyorBeltAction]:
        return FunctionalSpace(
            contains_fn=lambda x: isinstance(x, ConveyorBeltAction),
            sample_fn=lambda rng: ConveyorBeltAction(drop_package=bool(rng.choice(2))),
        )

    def _get_obs(self) -> ConveyorBeltState:
        assert self._current_state is not None
        return self._current_state

    def get_initial_states(self) -> EnumSpace[ConveyorBeltState]:
        return EnumSpace(
            [
                ConveyorBeltState(
                    positions=np.array([]),  # empty conveyor belt to start
                    falling_heights=np.array([]),
                    step_count=0,
                )
            ]
        )

    def get_next_states(
        self, state: ConveyorBeltState, action: ConveyorBeltAction
    ) -> EnumSpace[ConveyorBeltState]:
        # Get values from scene spec
        dt = self.scene_spec.dt
        gravity = self.scene_spec.gravity
        belt_length = self.scene_spec.belt_length
        box_width = self.scene_spec.box_width
        drop_start_height = self.scene_spec.drop_start_height
        initial_drop_position = self.scene_spec.initial_drop_position

        positions = list(state.positions)
        falling_heights = list(state.falling_heights)

        # Update falling heights
        for i in range(len(falling_heights)):
            if falling_heights[i] > 0:
                falling_heights[i] -= gravity * dt
                if falling_heights[i] <= 0:
                    falling_heights[i] = 0.0

        # Predict positions for non-falling boxes
        predicted_positions = []
        for p, h in zip(positions, falling_heights, strict=True):
            if h > 0.0:
                predicted_positions.append(p)
            else:
                v = self.scene_spec.conveyor_belt_velocity
                predicted_positions.append(p + v * dt)

        # Drop a new package if the action is drop
        if action.drop_package:
            # Create a new package!
            predicted_positions.append(initial_drop_position)
            falling_heights.append(drop_start_height)

        # Remove boxes that have fallen off the belt
        keep_pos, keep_fall = [], []
        for p, h in zip(predicted_positions, falling_heights, strict=True):
            if (p >= belt_length - box_width) or (p < 0):
                continue
            keep_pos.append(p)
            keep_fall.append(h)

        return EnumSpace(
            [
                ConveyorBeltState(
                    positions=np.array(keep_pos, dtype=np.float32),
                    falling_heights=np.array(keep_fall, dtype=np.float32),
                    step_count=state.step_count + 1,
                )
            ]
        )

    def actions_are_equal(
        self, action1: ConveyorBeltAction, action2: ConveyorBeltAction
    ) -> bool:
        return action1.drop_package == action2.drop_package

    def _get_reward_and_termination(
        self,
        state: ConveyorBeltState,
        action: ConveyorBeltAction,
        next_state: ConveyorBeltState,
    ) -> tuple[float, bool]:
        return 0.0, False

    def _render_state(
        self, state: ConveyorBeltState
    ) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode != "rgb_array":
            raise NotImplementedError(f"Unsupported render mode: {self.render_mode}")

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.set_xlim(0, self.scene_spec.belt_length)
        ax.set_ylim(0, 2)
        ax.axis("off")

        # Draw a red rectangle to visualize belt boundaries
        ax.add_patch(
            plt.Rectangle(
                (0, 0),
                self.scene_spec.belt_length,
                2,
                fill=False,
                edgecolor="red",
                linewidth=2,
                zorder=10,
            )
        )

        ax.add_patch(
            plt.Rectangle((0, 0.4), self.scene_spec.belt_length, 0.3, color="#777777")
        )
        ax.set_title(f"Conveyor Belt State: {len(state.positions)} Boxes")

        belt_y = 0.3
        belt_height = 0.4
        belt_length = self.scene_spec.belt_length
        y_pos = belt_y + belt_height

        belt_base = plt.Rectangle(
            (0, belt_y), belt_length, belt_height, color="#555555", zorder=0
        )
        ax.add_patch(belt_base)

        top_coords = [
            (0, belt_y + belt_height),
            (0.3, belt_y + belt_height + 0.15),
            (belt_length + 0.3, belt_y + belt_height + 0.15),
            (belt_length, belt_y + belt_height),
        ]
        belt_top = plt.Polygon(top_coords, color="#777777", zorder=2)
        ax.add_patch(belt_top)

        side_coords = [
            (belt_length, belt_y),
            (belt_length + 0.3, belt_y + 0.15),
            (belt_length + 0.3, belt_y + belt_height + 0.15),
            (belt_length, belt_y + belt_height),
        ]
        belt_side = plt.Polygon(side_coords, color="#444444", zorder=1)
        ax.add_patch(belt_side)

        roller_radius = 0.05
        roller_spacing = 1.0
        roller_count = int(np.ceil(belt_length / roller_spacing)) + 1
        time_elapsed = state.step_count * self.scene_spec.dt
        angular_speed = 2 * np.pi
        angle = -(time_elapsed * angular_speed) % (2 * np.pi)

        for i in range(roller_count):
            roller_x = i * roller_spacing - 0.1
            roller_y = belt_y - roller_radius
            roller = plt.Circle(
                (roller_x, roller_y), roller_radius, color="#222222", zorder=0
            )
            ax.add_patch(roller)

            spoke_length = roller_radius * 0.7
            axle_color = "#AAAAAA"

            x1_start = roller_x - spoke_length * np.cos(angle)
            y1_start = roller_y - spoke_length * np.sin(angle)
            x1_end = roller_x + spoke_length * np.cos(angle)
            y1_end = roller_y + spoke_length * np.sin(angle)

            angle_perp = angle + np.pi / 2
            x2_start = roller_x - spoke_length * np.cos(angle_perp)
            y2_start = roller_y - spoke_length * np.sin(angle_perp)
            x2_end = roller_x + spoke_length * np.cos(angle_perp)
            y2_end = roller_y + spoke_length * np.sin(angle_perp)

            ax.plot(
                [x1_start, x1_end],
                [y1_start, y1_end],
                color=axle_color,
                zorder=1,
                lw=0.5,
            )
            ax.plot(
                [x2_start, x2_end],
                [y2_start, y2_end],
                color=axle_color,
                zorder=1,
                lw=0.5,
            )

        def draw_3d_box(x: float, height: float, value: float | None = None):
            width, box_height = self.scene_spec.box_width, self.scene_spec.box_height
            color_min, color_max = -1.5, 1.5
            if value is not None:
                normalized = np.clip(
                    (value - color_min) / (color_max - color_min), 0, 1
                )
                base_color = plt.get_cmap("viridis")(normalized)
            else:
                base_color = plt.get_cmap("viridis")(0.5)

            top_rgb = np.minimum(np.array(base_color[:3]) * 1.3, 1.0)
            side_rgb = np.array(base_color[:3]) * 0.7
            top_color = (*top_rgb, 1.0)
            side_color = (*side_rgb, 1.0)

            y = y_pos + height
            front = plt.Rectangle((x, y), width, box_height, color=base_color, zorder=3)
            ax.add_patch(front)

            top_face = plt.Polygon(
                [
                    (x, y + box_height),
                    (x + 0.15, y + box_height + 0.15),
                    (x + width + 0.15, y + box_height + 0.15),
                    (x + width, y + box_height),
                ],
                color=top_color,
                zorder=4,
            )
            ax.add_patch(top_face)

            side_face = plt.Polygon(
                [
                    (x + width, y),
                    (x + width + 0.15, y + 0.15),
                    (x + width + 0.15, y + box_height + 0.15),
                    (x + width, y + box_height),
                ],
                color=side_color,
                zorder=2,
            )
            ax.add_patch(side_face)

            if value is not None:
                ax.text(
                    x + width / 2,
                    y + box_height / 2,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.5 else "black",
                    zorder=5,
                )

        for pos, height in zip(
            state.positions,
            state.falling_heights,
            strict=True,
        ):
            if pos < 0 or pos > self.scene_spec.belt_length:
                print(f"  Skipping box out of belt bounds: pos={pos}")
                continue
            draw_3d_box(pos, height)

        plt.tight_layout()
        image = fig2data(fig)
        plt.close(fig)

        return image  # type: ignore
