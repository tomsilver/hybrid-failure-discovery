"""PyBullet block stacking environment."""

from __future__ import annotations
from dataclasses import dataclass, field

from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.core import RenderFrame
from numpy.typing import NDArray
from tomsutils.spaces import EnumSpace, FunctionalSpace
from tomsutils.utils import fig2data

from hybrid_failure_discovery.envs.constraint_based_env_model import (
    ConstraintBasedGymEnv,
)

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from numpy.typing import NDArray
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block


@dataclass(frozen=True)
class BlockState:
    """A state of a single block."""

    id: int
    pose: Pose


@dataclass(frozen=True)
class RobotState:
    """A state of the robot."""

    joint_positions: JointPositions


@dataclass(frozen=True)
class BlocksEnvState:
    """A state in the blocks environment."""

    robot: RobotState
    blocks: list[BlockState]
    held_block_id: int | None = None
    held_block_grasp: Pose | None = None


@dataclass(frozen=True)
class BlocksAction:
    """An action in the blocks environment."""

    robot_arm_joint_delta: JointPositions
    gripper_action: int  # -1 for close, 0 for no change, 1 for open


@dataclass(frozen=True)
class BlocksEnvSceneSpec:
    """Static hyperparameters for the blocks environment."""

    gravity: float = 9.80665
    num_sim_steps_per_step: int = 30

    robot_name: str = "panda"
    robot_base_pose: Pose = Pose.identity()
    initial_joints: JointPositions = field(
        default_factory=lambda: [
            -1.6760817784086874,
            -0.8633617886115512,
            1.0820023618960484,
            -1.7862427129376002,
            0.7563762599673787,
            1.3595324116603988,
            1.7604148617061273,
            0.04,
            0.04,
        ]
    )
    robot_max_joint_delta: float = np.inf

    robot_stand_pose: Pose = Pose((0.0, 0.0, -0.2))
    robot_stand_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    robot_stand_half_extents: tuple[float, float, float] = (0.2, 0.2, 0.225)

    table_pose: Pose = Pose((0.5, 0.0, -0.175))
    table_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    table_half_extents: tuple[float, float, float] = (0.25, 0.4, 0.25)

    block_rgba: tuple[float, float, float, float] = (0.5, 0.0, 0.5, 1.0)
    block_text_rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    block_half_extents: tuple[float, float, float] = (0.025, 0.025, 0.025)
    block_mass: float = 0.5
    block_friction: float = 0.9
    num_blocks: int = 6

    @property
    def block_init_position_lower(self) -> tuple[float, float, float]:
        """Lower bounds for block position."""
        return (
            self.table_pose.position[0]
            - self.table_half_extents[0]
            + self.block_half_extents[0],
            self.table_pose.position[1]
            - self.table_half_extents[1]
            + self.block_half_extents[1],
            self.table_pose.position[2]
            + self.table_half_extents[2]
            + self.block_half_extents[2],
        )

    @property
    def block_init_position_upper(self) -> tuple[float, float, float]:
        """Upper bounds for block position."""
        return (
            self.table_pose.position[0]
            + self.table_half_extents[0]
            - self.block_half_extents[0],
            self.table_pose.position[1]
            + self.table_half_extents[1]
            - self.block_half_extents[1],
            self.table_pose.position[2]
            + self.table_half_extents[2]
            + self.block_half_extents[2],
        )

    def get_camera_kwargs(self) -> dict[str, Any]:
        """Derived kwargs for taking images."""
        return {
            "camera_target": self.robot_base_pose.position,
            "camera_yaw": 90,
            "camera_distance": 1.5,
            "camera_pitch": -20,
        }


class BlocksEnv(ConstraintBasedGymEnv[BlocksEnvState, BlocksAction]):
    """A pybullet blocks environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        scene_spec: BlocksEnvSceneSpec = BlocksEnvSceneSpec(),
        seed: int = 0,
        use_gui: bool = False,
    ) -> None:
        self.scene_spec = scene_spec
        self.render_mode = "rgb_array"
        super().__init__(seed)

        # Create the PyBullet client.
        if use_gui:
            self.physics_client_id = create_gui_connection(camera_yaw=180)
        else:
            self.physics_client_id = p.connect(p.DIRECT)

        # Set gravity.
        p.setGravity(
            0,
            0,
            -self.scene_spec.gravity,
            physicsClientId=self.physics_client_id,
        )

        # Create robot.
        robot = create_pybullet_robot(
            self.scene_spec.robot_name,
            self.physics_client_id,
            base_pose=self.scene_spec.robot_base_pose,
            control_mode="reset",
            home_joint_positions=self.scene_spec.initial_joints,
        )
        assert isinstance(robot, FingeredSingleArmPyBulletRobot)
        robot.close_fingers()
        self.robot = robot

        # Create robot stand.
        self.robot_stand_id = create_pybullet_block(
            self.scene_spec.robot_stand_rgba,
            half_extents=self.scene_spec.robot_stand_half_extents,
            physics_client_id=self.physics_client_id,
        )
        set_pose(
            self.robot_stand_id,
            self.scene_spec.robot_stand_pose,
            self.physics_client_id,
        )

        # Create table.
        self.table_id = create_pybullet_block(
            self.scene_spec.table_rgba,
            half_extents=self.scene_spec.table_half_extents,
            physics_client_id=self.physics_client_id,
        )
        set_pose(
            self.table_id, self.scene_spec.table_pose, self.physics_client_id
        )

        # Create the blocks. Their poses will be reset later.
        self.block_ids = [
            create_pybullet_block(self.scene_spec.block_rgba,
                                  self.scene_spec.block_half_extents,
                                  self.physics_client_id,
                                  self.scene_spec.block_mass,
                                  self.scene_spec.block_friction)
            for _ in range(self.scene_spec.num_blocks)
        ]

        # Initialize the grasp.
        self.current_grasp_transform: Pose | None = None
        self.current_held_object_id: int | None = None        

    def _create_action_space(self) -> FunctionalSpace[BlocksAction]:
        return FunctionalSpace(
            contains_fn=lambda x: isinstance(x, BlocksAction),
        )

    def _get_obs(self) -> BlocksEnvState:
        return self._get_state()

    def _get_state(self) -> BlocksEnvState:
        # Get the robot state.
        robot_state = RobotState(self.robot.get_joint_positions())

        # Get the block states.
        block_states = []
        for block_id in self.block_ids:
            block_state = BlockState(block_id, get_pose(block_id, self.physics_client_id))
            block_states.append(block_state)

        return BlocksEnvState(robot_state, block_states, self.current_held_object_id, self.current_grasp_transform)

    def get_initial_states(self) -> EnumSpace[BlocksEnvState]:
        # Fully constrained for now.

        # Reset the robot.
        self.robot.set_joints(self.scene_spec.initial_joints)

        # Reset the grasp.
        self.current_grasp_transform = None
        self.current_held_object_id = None

        # Reset the blocks in just a straight line on the table.
        lx, ly, lz = self.scene_spec.block_init_position_lower
        ux, uy, uz = self.scene_spec.block_init_position_upper
        x = np.mean([lx, ux])
        z = np.mean([lz, uz])
        for i, y in enumerate(np.linspace(ly, uy, num=self.scene_spec.num_blocks, endpoint=True)):
            block_id = self.block_ids[i]
            pose = Pose((x, y, z))
            set_pose(block_id, pose, self.physics_client_id)

        state = self._get_state()

        return EnumSpace([state])

    def get_next_states(
        self, state: BlocksEnvState, action: BlocksAction
    ) -> EnumSpace[BlocksEnvState]:
        # TODO
        return EnumSpace([state])

    def _get_reward_and_termination(
        self, state: BlocksEnvState, action: BlocksAction, next_state: BlocksEnvState
    ) -> tuple[float, bool]:
        return 0.0, False

    def _render_state(self, state: ObsType) -> RenderFrame | list[RenderFrame] | None:
        return capture_image(
            self.physics_client_id,
            **self.scene_spec.get_camera_kwargs()
        )
