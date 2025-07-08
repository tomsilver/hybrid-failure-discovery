"""This module defines the ConveyorBeltEnv environment for simulating a 1D
conveyor belt with loading and shifting actions."""

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
    """Represents the current state of the conveyor belt, the values of boxes
    1 - 6 display the current velocity of the conveyor belt (and consequently
    simultaneous speed of all the boxes) reflective of the action taken."""

    values: NDArray[np.float32]


@dataclass(frozen=True)
class ConveyorBeltAction:
    """Represents a discrete action taken in the conveyor belt environment.

    Where an action of index 0 represents moving in reverse (-1.0 m/s),
    an action of index 1 represents no update or action taken, an action
    of index 2 represents moving forward slowly (0.5 m/s),  an action of
    index 3 represents moving forward at a normal pace (1.0 m/s), an
    action of index 4 represents moving forward fast (1.5 m/s).
    """

    index: int  # Discrete action index from 0 to 2 representing directionality taken


@dataclass(frozen=True)
class ConveyorBeltSceneSpec:
    """Setting values for the boxes that initially represent the normal speed
    of the conveyer belt (1.0 m/s)"""

    init_box1: float = 1.0
    init_box2: float = 1.0
    init_box3: float = 1.0
    init_box4: float = 1.0
    init_box5: float = 1.0
    init_box6: float = 1.0


class ConveyorBeltEnv(ConstraintBasedGymEnv[ConveyorBeltState, ConveyorBeltAction]):
    """A simplified 1D conveyor belt environment with load/unload behavior and
    3 actions."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 1.5}  # slowed down fps

    def __init__(
        self,
        render_mode: str = "rgb_array",
        scene_spec: ConveyorBeltSceneSpec = ConveyorBeltSceneSpec(),
        seed: int = 0,
    ) -> None:
        self.scene_spec = scene_spec
        self.render_mode = render_mode
        self.observation_dim = 6

        self._last_inserted_value: float | None = None
        self.action_dim = 5  # Only actions 0, 1, 2, 3, 4 are valid
        super().__init__(seed)
        self._rng = np.random.default_rng(seed)

        self._step_count = 0

    def _create_action_space(self) -> FunctionalSpace[ConveyorBeltAction]:
        return FunctionalSpace(
            contains_fn=lambda x: isinstance(x, ConveyorBeltAction),
            sample_fn=lambda rng: ConveyorBeltAction(
                index=rng.integers(0, self.action_dim)
            ),
        )

    def _get_obs(self) -> ConveyorBeltState:
        assert self._current_state is not None
        return self._current_state

    def get_initial_states(
        self,
    ) -> EnumSpace[ConveyorBeltState]:
        values = np.array(
            [
                self.scene_spec.init_box1,
                self.scene_spec.init_box2,
                self.scene_spec.init_box3,
                self.scene_spec.init_box4,
                self.scene_spec.init_box5,
                self.scene_spec.init_box6,
            ],
            dtype=np.float32,
        )
        state = ConveyorBeltState(values)
        return EnumSpace([state])

    def get_next_states(
        self, state: ConveyorBeltState, action: ConveyorBeltAction
    ) -> EnumSpace[ConveyorBeltState]:
        assert self._np_random is not None

        if action.index == 0:  # Reverse
            new_value = -1.0
        elif action.index == 1:  # Stop
            new_value = 0.0
        elif action.index == 2:  # Slow Forward
            new_value = 0.5
        elif action.index == 3:  # Normal Speed
            new_value = 1.0
        elif action.index == 4:  # Fast
            new_value = 1.5
        else:
            raise ValueError(f"Invalid action index: {action.index}")

        # Set all values to the selected speed
        next_values = np.full_like(state.values, fill_value=new_value, dtype=np.float32)

        self._last_inserted_value = new_value
        next_state = ConveyorBeltState(values=next_values)
        return EnumSpace([next_state])

    def actions_are_equal(
        self, action1: ConveyorBeltAction, action2: ConveyorBeltAction
    ) -> bool:
        return action1.index == action2.index

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
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 2)
        ax.axis("off")
        ax.set_title("Conveyor Belt State: 6 Boxes")

        box_count = 6
        box_width = 0.5
        spacing = 0.25
        box_height = 0.5
        belt_y = 0.3
        belt_height = 0.4
        y_pos = belt_y + belt_height

        belt_base = plt.Rectangle(
            (0, belt_y), 5, belt_height, color="#555555", zorder=0
        )
        ax.add_patch(belt_base)

        top_coords = [
            (0, belt_y + belt_height),
            (0.3, belt_y + belt_height + 0.15),
            (5.3, belt_y + belt_height + 0.15),
            (5, belt_y + belt_height),
        ]
        belt_top = plt.Polygon(top_coords, color="#777777", zorder=2)
        ax.add_patch(belt_top)

        side_coords = [
            (5, belt_y),
            (5.3, belt_y + 0.15),
            (5.3, belt_y + belt_height + 0.15),
            (5, belt_y + belt_height),
        ]
        belt_side = plt.Polygon(side_coords, color="#444444", zorder=1)
        ax.add_patch(belt_side)

        roller_radius = 0.05
        roller_spacing = 1.0
        roller_count = 6
        angle = -(self._step_count * 0.15) % (2 * np.pi)

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
                linewidth=2,
                zorder=3,
            )
            ax.plot(
                [x2_start, x2_end],
                [y2_start, y2_end],
                color=axle_color,
                linewidth=2,
                zorder=3,
            )

        def draw_3d_box(
            x: float, y: float, width: float, height: float, value: float
        ) -> None:
            color_min = -1.5
            color_max = 1.5
            normalized_value = (value - color_min) / (color_max - color_min)
            # Clamp to [0,1]
            normalized_value = np.clip(normalized_value, 0.0, 1.0)

            base_color = plt.get_cmap("viridis")(normalized_value)
            top_rgb = np.minimum(np.array(base_color[:3]) * 1.3, 1.0)
            top_color = (*top_rgb, base_color[3])
            side_rgb = np.array(base_color[:3]) * 0.7
            side_color = (*side_rgb, base_color[3])

            front = plt.Rectangle((x, y), width, height, color=base_color, zorder=3)
            ax.add_patch(front)

            top_coords = [
                (x, y + height),
                (x + 0.15, y + height + 0.15),
                (x + width + 0.15, y + height + 0.15),
                (x + width, y + height),
            ]
            top_face = plt.Polygon(top_coords, color=top_color, zorder=4)
            ax.add_patch(top_face)

            side_coords = [
                (x + width, y),
                (x + width + 0.15, y + 0.15),
                (x + width + 0.15, y + height + 0.15),
                (x + width, y + height),
            ]
            side_face = plt.Polygon(side_coords, color=side_color, zorder=2)
            ax.add_patch(side_face)

            ax.text(
                x + width / 2,
                y + height / 2,
                f"{value:.2f}",
                color="white" if value > 0.5 else "black",
                fontsize=12,
                ha="center",
                va="center",
                zorder=5,
            )

        total_width = box_count * box_width + (box_count - 1) * spacing
        left_margin = (5 - total_width) / 2

        for i in range(box_count):
            val = state.values[i] if i < len(state.values) else 0.0
            x_pos = left_margin + i * (box_width + spacing)
            draw_3d_box(x_pos, y_pos, box_width, box_height, val)

        plt.tight_layout()
        image = fig2data(fig)
        plt.close(fig)

        self._step_count += 1
        return image  # type: ignore
