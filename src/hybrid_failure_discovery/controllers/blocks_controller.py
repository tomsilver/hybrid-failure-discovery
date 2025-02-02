"""A controller for the blocks environment."""

import abc
from functools import partial

import numpy as np
from gymnasium.spaces import Space
from pybullet_helpers.geometry import Pose, iter_between_poses
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    smoothly_follow_end_effector_path,
)
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from tomsutils.spaces import FunctionalSpace

from hybrid_failure_discovery.controllers.controller import ConstraintBasedController
from hybrid_failure_discovery.envs.blocks_env import (
    BlocksAction,
    BlocksEnv,
    BlocksEnvSceneSpec,
    BlocksEnvState,
)


class BlocksController(ConstraintBasedController[BlocksEnvState, BlocksAction]):
    """A blocks env controller that randomly picks and places blocks."""

    def __init__(
        self,
        seed: int,
        scene_spec: BlocksEnvSceneSpec,
        safe_height: float = 0.25,
    ) -> None:
        super().__init__(seed)
        self._scene_spec = scene_spec

        # Create a simulated robot for kinematics and such.
        self._sim_robot = BlocksEnv(scene_spec, seed=seed).robot

        # Create options.
        pick_options = [
            PickBlockOption(f"block{i}", seed, self._sim_robot, scene_spec, safe_height)
            for i in range(scene_spec.num_blocks)
        ]
        stack_options = [
            StackBlockOption(
                f"block{i}", seed, self._sim_robot, scene_spec, safe_height
            )
            for i in range(scene_spec.num_blocks)
        ]

        self._options = pick_options + stack_options

        # Track the current option.
        self._current_option: BlocksOption | None = None

    def reset(self, initial_state: BlocksEnvState) -> None:
        self._current_option = None

    def step_action_space(self, state: BlocksEnvState) -> Space[BlocksAction]:
        return FunctionalSpace(
            contains_fn=lambda x: isinstance(x, BlocksAction),
            sample_fn=partial(self._sample_action, state),
        )

    def _sample_action(
        self, state: BlocksEnvState, rng: np.random.Generator
    ) -> BlocksAction:
        if self._current_option is None or self._current_option.terminate(state):
            # Sample a new initiable option.
            self._current_option = None
            idxs = list(range(len(self._options)))
            rng.shuffle(idxs)
            ordered_options = [self._options[i] for i in idxs]
            for option in ordered_options:
                if option.can_initiate(state):
                    self._current_option = option
                    self._current_option.initiate(state)
                    break
            assert self._current_option is not None
        return self._current_option.step(state)


class BlocksOption:
    """A partial controller that initiates and then terminates, e.g., pick."""

    def __init__(
        self,
        seed: int,
        robot: FingeredSingleArmPyBulletRobot,
        scene_spec: BlocksEnvSceneSpec,
        safe_height: float,
    ) -> None:
        self._seed = seed
        self._robot = robot
        self._scene_spec = scene_spec
        self._safe_height = safe_height
        self._rng = np.random.default_rng(seed)
        self._plan: list[BlocksAction] = []
        self._joint_distance_fn = create_joint_distance_fn(self._robot)

    @abc.abstractmethod
    def can_initiate(self, state: BlocksEnvState) -> bool:
        """Whether the option could be initiated in the given state."""

    @abc.abstractmethod
    def initiate(self, state: BlocksEnvState) -> None:
        """Initiate the option."""

    def step(self, state: BlocksEnvState) -> BlocksAction:
        """Step the option."""
        del state  # not used right now
        return self._plan.pop(0)

    def terminate(self, state) -> bool:
        """Check for termination."""
        del state  # not used right now
        return not self._plan

    def _motion_plan_to_plan(
        self, motion_plan: list[JointPositions]
    ) -> list[BlocksAction]:
        plan = []
        for t in range(len(motion_plan) - 1):
            delta = np.subtract(motion_plan[t + 1], motion_plan[t]).tolist()[:7]
            action = BlocksAction(delta, gripper_action=0)
            plan.append(action)
        return plan


class PickBlockOption(BlocksOption):
    """A partial controller for picking an exposed block."""

    def __init__(self, block_name: str, *args, **kwargs) -> None:
        self._block_name = block_name
        super().__init__(*args, **kwargs)

    def can_initiate(self, state: BlocksEnvState) -> bool:
        # Robot hand must be empty.
        if state.held_block_name:
            return False

        # The block must not have anything on top of it.
        return _block_has_nothing_above(self._block_name, state, self._scene_spec)

    def initiate(self, state: BlocksEnvState) -> None:
        # Reset the simulated robot to the given state.
        self._robot.set_joints(state.robot.joint_positions)

        start_pose = self._robot.get_end_effector_pose()
        block_pose = state.get_block_state(self._block_name).pose
        waypoint1 = Pose(
            (start_pose.position[0], start_pose.position[1], self._safe_height),
            start_pose.orientation,
        )
        waypoint2 = Pose(
            (block_pose.position[0], block_pose.position[1], self._safe_height),
            start_pose.orientation,
        )
        waypoint3 = Pose(block_pose.position, start_pose.orientation)

        waypoints = [
            start_pose,
            waypoint1,
            waypoint2,
            waypoint3,
        ]

        end_effector_path: list[Pose] = []
        for p1, p2 in zip(waypoints[:-1], waypoints[1:], strict=True):
            end_effector_path.extend(iter_between_poses(p1, p2))

        motion_plan = smoothly_follow_end_effector_path(
            self._robot,
            end_effector_path,
            state.robot.joint_positions,
            collision_ids=set(),
            joint_distance_fn=self._joint_distance_fn,
            max_time=0.5,
        )

        self._plan.append(BlocksAction([0.0] * 7, gripper_action=1))  # open
        self._plan = self._motion_plan_to_plan(motion_plan)
        self._plan.append(BlocksAction([0.0] * 7, gripper_action=-1))  # close


class StackBlockOption(BlocksOption):
    """A partial controller for stacking a held block on an exposed block."""

    def __init__(self, block_name: str, *args, **kwargs) -> None:
        self._block_name = block_name  # the exposed block
        super().__init__(*args, **kwargs)

    def can_initiate(self, state: BlocksEnvState) -> bool:
        # Robot must be holding some block, but not the target.
        if state.held_block_name in [None, self._block_name]:
            return False

        # The block must not have anything on top of it.
        return _block_has_nothing_above(self._block_name, state, self._scene_spec)

    def initiate(self, state: BlocksEnvState) -> None:
        # Reset the simulated robot to the given state.
        self._robot.set_joints(state.robot.joint_positions)

        start_pose = self._robot.get_end_effector_pose()
        block_pose = state.get_block_state(self._block_name).pose
        waypoint1 = Pose(
            (start_pose.position[0], start_pose.position[1], self._safe_height),
            start_pose.orientation,
        )
        waypoint2 = Pose(
            (block_pose.position[0], block_pose.position[1], self._safe_height),
            start_pose.orientation,
        )
        waypoint3 = Pose(
            (
                block_pose.position[0],
                block_pose.position[1],
                block_pose.position[2] + 2 * self._scene_spec.block_half_extents[2],
            ),
            start_pose.orientation,
        )

        waypoints = [
            start_pose,
            waypoint1,
            waypoint2,
            waypoint3,
        ]

        end_effector_path: list[Pose] = []
        for p1, p2 in zip(waypoints[:-1], waypoints[1:], strict=True):
            end_effector_path.extend(iter_between_poses(p1, p2))

        motion_plan = smoothly_follow_end_effector_path(
            self._robot,
            end_effector_path,
            state.robot.joint_positions,
            collision_ids=set(),
            joint_distance_fn=self._joint_distance_fn,
            max_time=0.5,
        )

        self._plan = self._motion_plan_to_plan(motion_plan)
        self._plan.append(BlocksAction([0.0] * 7, gripper_action=1))  # open


def _block_has_nothing_above(
    block_name: str, state: BlocksEnvState, scene_spec: BlocksEnvSceneSpec
) -> bool:
    block_state = state.get_block_state(block_name)
    x_thresh = scene_spec.block_half_extents[0]
    y_thresh = scene_spec.block_half_extents[1]
    for other_block_state in state.blocks:
        if other_block_state.name == block_name:
            continue
        # Check if this other block is above the main one.
        if other_block_state.pose.position[2] <= block_state.pose.position[2]:
            continue
        # Check if close enough in the x plane.
        if (
            abs(other_block_state.pose.position[0] - block_state.pose.position[0])
            > x_thresh
        ):
            continue
        # Check if close enough in the y plane.
        if (
            abs(other_block_state.pose.position[1] - block_state.pose.position[1])
            > y_thresh
        ):
            continue
        # On is true.
        return False
    return True
