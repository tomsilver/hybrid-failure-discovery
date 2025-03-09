"""A controller for the blocks environment."""

import abc
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from gymnasium.spaces import Space
from pybullet_helpers.geometry import Pose, get_pose, iter_between_poses
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    smoothly_follow_end_effector_path,
)
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from relational_structs import GroundAtom, Object, Predicate, Type
from task_then_motion_planning.structs import Perceiver
from tomsutils.spaces import EnumSpace, FunctionalSpace

from hybrid_failure_discovery.controllers.controller import ConstraintBasedController
from hybrid_failure_discovery.envs.blocks_env import (
    BlocksAction,
    BlocksEnv,
    BlocksEnvSceneSpec,
    BlocksEnvState,
)

################################################################################
#                                Commands                                      #
################################################################################


@dataclass(frozen=True)
class BlocksCommand:
    """A command in the blocks environment."""

    towers: list[list[str]]  # target towers to build


################################################################################
#                              "Perception"                                    #
################################################################################

# Create generic types.
robot_type = Type("robot")
object_type = Type("obj")  # NOTE: pyperplan breaks with 'object' type name
TYPES = {robot_type, object_type}

# Create predicates.
IsMovable = Predicate("IsMovable", [object_type])
NotIsMovable = Predicate("NotIsMovable", [object_type])
On = Predicate("On", [object_type, object_type])
NothingOn = Predicate("NothingOn", [object_type])
Holding = Predicate("Holding", [robot_type, object_type])
GripperEmpty = Predicate("GripperEmpty", [robot_type])
Clear = Predicate("Clear", [object_type])
PREDICATES = {IsMovable, NotIsMovable, On, NothingOn, Holding, GripperEmpty, Clear}


class PyBulletBlocksPerceiver(Perceiver[BlocksEnvState]):
    """A perceiver for blocks environment."""

    def __init__(self, sim: BlocksEnv) -> None:
        # Use the simulator for geometric computations.
        self._sim = sim

        # Create objects.
        self._robot = Object("robot", robot_type)
        self._table = Object("table", object_type)
        self._blocks = [
            Object(f"block{i}", object_type) for i in range(sim.scene_spec.num_blocks)
        ]

        # Map from symbolic objects to PyBullet IDs in simulator.
        self._pybullet_ids = {
            self._robot: self._sim.robot.robot_id,
            self._table: self._sim.table_id,
        }
        for block, block_id in zip(self._blocks, sim.block_ids, strict=True):
            self._pybullet_ids[block] = block_id

        # Store on relations for predicate interpretations.
        self._on_relations: set[tuple[Object, Object]] = set()

        # Create predicate interpreters.
        self._predicate_interpreters = [
            self._interpret_IsMovable,
            self._interpret_NotIsMovable,
            self._interpret_On,
            self._interpret_NothingOn,
            self._interpret_Holding,
            self._interpret_GripperEmpty,
            self._interpret_Clear,
        ]

    def reset(
        self,
        obs: BlocksEnvState,
        info: dict[str, Any],
    ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        atoms = self._parse_observation(obs)
        objects = self._get_objects()
        goal = self._get_goal(info)
        return objects, atoms, goal

    def step(self, obs: BlocksEnvState) -> set[GroundAtom]:
        atoms = self._parse_observation(obs)
        return atoms

    def _get_objects(self) -> set[Object]:
        return set(self._pybullet_ids)

    def _get_goal(self, info: dict[str, Any]) -> set[GroundAtom]:
        assert "goal" in info and isinstance(info["goal"], BlocksCommand)
        name_to_obj = {b.name: b for b in self._blocks}
        goal: set[GroundAtom] = set()
        for pile in info["goal"].towers:
            for bottom_letter, top_letter in zip(pile[:-1], pile[1:], strict=True):
                top = name_to_obj[top_letter]
                bottom = name_to_obj[bottom_letter]
                atom = GroundAtom(On, [top, bottom])
                goal.add(atom)
        return goal

    def _parse_observation(self, obs: BlocksEnvState) -> set[GroundAtom]:
        # Sync the simulator so that interpretation functions can use PyBullet
        # direction.
        self._sim.set_state(obs)

        # Compute which things are on which other things.
        self._on_relations = self._get_on_relations_from_sim()

        # Create current atoms.
        atoms: set[GroundAtom] = set()
        for interpret_fn in self._predicate_interpreters:
            atoms.update(interpret_fn())

        return atoms

    def _get_on_relations_from_sim(self) -> set[tuple[Object, Object]]:
        on_relations = set()
        candidates = {o for o in self._get_objects() if o.is_instance(object_type)}
        for obj1 in candidates:
            obj1_pybullet_id = self._pybullet_ids[obj1]
            pose1 = get_pose(obj1_pybullet_id, self._sim.physics_client_id)
            for obj2 in candidates:
                if obj1 == obj2:
                    continue
                obj2_pybullet_id = self._pybullet_ids[obj2]
                # Check if obj1 pose is above obj2 pose.
                pose2 = get_pose(obj2_pybullet_id, self._sim.physics_client_id)
                if pose1.position[2] < pose2.position[2]:
                    continue
                # Check for contact.
                if check_body_collisions(
                    obj1_pybullet_id,
                    obj2_pybullet_id,
                    self._sim.physics_client_id,
                    distance_threshold=1e-3,
                ):
                    on_relations.add((obj1, obj2))
        return on_relations

    def _interpret_IsMovable(self) -> set[GroundAtom]:
        return {GroundAtom(IsMovable, [o]) for o in self._blocks}

    def _interpret_NotIsMovable(self) -> set[GroundAtom]:
        objs = {o for o in self._get_objects() if o.is_instance(object_type)}
        movable_atoms = self._interpret_IsMovable()
        movable_objs = {a.objects[0] for a in movable_atoms}
        not_movable_objs = objs - movable_objs
        return {GroundAtom(NotIsMovable, [o]) for o in not_movable_objs}

    def _interpret_On(self) -> set[GroundAtom]:
        return {GroundAtom(On, r) for r in self._on_relations}

    def _interpret_NothingOn(self) -> set[GroundAtom]:
        objs = {o for o in self._get_objects() if o.is_instance(object_type)}
        for _, bot in self._on_relations:
            objs.discard(bot)
        return {GroundAtom(NothingOn, [o]) for o in objs}

    def _interpret_Holding(self) -> set[GroundAtom]:
        if self._sim.current_held_block is not None:
            name = self._sim.current_held_block
            matches = [b for b in self._blocks if b.name == name]
            assert len(matches) == 1
            held_obj = matches[0]
            return {GroundAtom(Holding, [self._robot, held_obj])}
        return set()

    def _interpret_GripperEmpty(self) -> set[GroundAtom]:
        if not self._sim.current_grasp_transform:
            return {GroundAtom(GripperEmpty, [self._robot])}
        return set()

    def _interpret_Clear(self) -> set[GroundAtom]:
        clear_objects = {
            self._table
        }  # Table always clear since we can sample free spots
        nothing_on_atoms = self._interpret_NothingOn()
        clear_objects.update(atom.objects[0] for atom in nothing_on_atoms)
        return {GroundAtom(Clear, [obj]) for obj in clear_objects}


################################################################################
#                                Controller                                    #
################################################################################


class BlocksController(
    ConstraintBasedController[BlocksEnvState, BlocksAction, BlocksCommand]
):
    """A blocks env controller that randomly picks and places blocks."""

    def __init__(
        self,
        seed: int,
        scene_spec: BlocksEnvSceneSpec,
        safe_height: float = 0.25,
        max_smoothing_iters_per_step: int = 1,
    ) -> None:
        super().__init__(seed)
        self._scene_spec = scene_spec

        # Create a simulated robot for kinematics and such.
        self._sim_robot = BlocksEnv(scene_spec, seed=seed).robot

        # Create options.
        pick_options = [
            PickBlockOption(
                f"block{i}",
                seed,
                self._sim_robot,
                scene_spec,
                safe_height,
                max_smoothing_iters_per_step,
            )
            for i in range(scene_spec.num_blocks)
        ]
        stack_options = [
            StackBlockOption(
                f"block{i}",
                seed,
                self._sim_robot,
                scene_spec,
                safe_height,
                max_smoothing_iters_per_step,
            )
            for i in range(scene_spec.num_blocks)
        ]

        self._options = pick_options + stack_options

        # Track the current option.
        self._current_option: BlocksOption | None = None

    def reset(self, initial_state: BlocksEnvState) -> None:
        self._current_option = None
        self._np_random = np.random.default_rng(self._seed)
        for option in self._options:
            option.reset(initial_state)

    def step_action_space(
        self, state: BlocksEnvState, command: BlocksCommand
    ) -> Space[BlocksAction]:
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

    def get_command_space(self) -> Space[BlocksCommand]:
        return EnumSpace([BlocksCommand([])])


class BlocksOption:
    """A partial controller that initiates and then terminates, e.g., pick."""

    def __init__(
        self,
        seed: int,
        robot: FingeredSingleArmPyBulletRobot,
        scene_spec: BlocksEnvSceneSpec,
        safe_height: float,
        max_smoothing_iters_per_step: int = 1,
    ) -> None:
        self._seed = seed
        self._robot = robot
        self._scene_spec = scene_spec
        self._safe_height = safe_height
        self._max_smoothing_iters_per_step = max_smoothing_iters_per_step
        self._rng = np.random.default_rng(seed)
        self._plan: list[BlocksAction] = []
        self._joint_distance_fn = create_joint_distance_fn(self._robot)

    @abc.abstractmethod
    def can_initiate(self, state: BlocksEnvState) -> bool:
        """Whether the option could be initiated in the given state."""

    @abc.abstractmethod
    def initiate(self, state: BlocksEnvState) -> None:
        """Initiate the option."""

    def reset(self, initial_state: BlocksEnvState) -> None:
        """Reset the option."""
        del initial_state
        self._rng = np.random.default_rng(self._seed)

    def step(self, state: BlocksEnvState) -> BlocksAction:
        """Step the option."""
        del state  # not used right now
        act = self._plan.pop(0)
        return act

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
            max_smoothing_iters_per_step=self._max_smoothing_iters_per_step,
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
            max_smoothing_iters_per_step=self._max_smoothing_iters_per_step,
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
