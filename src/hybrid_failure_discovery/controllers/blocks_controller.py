"""A controller for the blocks environment."""

import abc
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Sequence

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
from relational_structs import (
    GroundAtom,
    GroundOperator,
    LiftedAtom,
    LiftedOperator,
    Object,
    Predicate,
    Type,
    Variable,
)
from task_then_motion_planning.planning import TaskThenMotionPlanner
from task_then_motion_planning.structs import LiftedOperatorSkill, Perceiver, Skill
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


class BlocksPerceiver(Perceiver[BlocksEnvState]):
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
        for block in self._blocks:
            self._pybullet_ids[block] = self._sim.block_ids[block.name]

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
        print(f"Reset perceiver with goal: {sorted(goal)}")
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
#                                 Operators                                    #
################################################################################

Robot = Variable("?robot", robot_type)
Obj = Variable("?obj", object_type)
Surface = Variable("?surface", object_type)

PickOperator = LiftedOperator(
    "Pick",
    [Robot, Obj, Surface],
    preconditions={
        LiftedAtom(IsMovable, [Obj]),
        LiftedAtom(NotIsMovable, [Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(NothingOn, [Obj]),
        LiftedAtom(On, [Obj, Surface]),
    },
    add_effects={
        LiftedAtom(Holding, [Robot, Obj]),
    },
    delete_effects={
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(On, [Obj, Surface]),
    },
)

PlaceOperator = LiftedOperator(
    "Place",
    [Robot, Obj, Surface],
    preconditions={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NotIsMovable, [Surface]),
    },
    add_effects={
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
    },
    delete_effects={
        LiftedAtom(Holding, [Robot, Obj]),
    },
)

UnstackOperator = LiftedOperator(
    "Unstack",
    [Robot, Obj, Surface],
    preconditions={
        LiftedAtom(IsMovable, [Obj]),
        LiftedAtom(IsMovable, [Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(NothingOn, [Obj]),
        LiftedAtom(On, [Obj, Surface]),
    },
    add_effects={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NothingOn, [Surface]),
    },
    delete_effects={
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(On, [Obj, Surface]),
    },
)

StackOperator = LiftedOperator(
    "Stack",
    [Robot, Obj, Surface],
    preconditions={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NothingOn, [Surface]),
        LiftedAtom(IsMovable, [Surface]),
    },
    add_effects={
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
    },
    delete_effects={
        LiftedAtom(NothingOn, [Surface]),
        LiftedAtom(Holding, [Robot, Obj]),
    },
)

OPERATORS = {
    PickOperator,
    PlaceOperator,
    UnstackOperator,
    StackOperator,
}

################################################################################
#                                  Skills                                      #
################################################################################


class BlocksSkill(LiftedOperatorSkill[BlocksEnvState, BlocksAction]):
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
        self._current_plan: list[BlocksAction] = []
        self._joint_distance_fn = create_joint_distance_fn(self._robot)
        super().__init__()

    def reset(self, ground_operator: GroundOperator) -> None:
        self._current_plan = []
        return super().reset(ground_operator)

    def _get_action_given_objects(
        self, objects: Sequence[Object], obs: BlocksEnvState
    ) -> BlocksAction:
        if not self._current_plan:
            self._current_plan = self._get_plan_given_objects(objects, obs)
        return self._current_plan.pop(0)

    @abc.abstractmethod
    def _get_plan_given_objects(
        self, objects: Sequence[Object], obs: BlocksEnvState
    ) -> list[BlocksAction]:
        """Run planning given the objects and state."""


def _motion_plan_to_plan(motion_plan: list[JointPositions]) -> list[BlocksAction]:
    plan = []
    for t in range(len(motion_plan) - 1):
        delta = np.subtract(motion_plan[t + 1], motion_plan[t]).tolist()[:7]
        action = BlocksAction(delta, gripper_action=0)
        plan.append(action)
    return plan


def _get_pick_block_plan(
    block_name: str,
    state: BlocksEnvState,
    robot: FingeredSingleArmPyBulletRobot,
    safe_height: float,
    joint_distance_fn: Callable[[JointPositions, JointPositions], float],
    max_smoothing_iters_per_step: int,
) -> list[BlocksAction]:
    # Reset the simulated robot to the given state.
    robot.set_joints(state.robot.joint_positions)

    start_pose = robot.get_end_effector_pose()
    block_pose = state.get_block_state(block_name).pose
    waypoint1 = Pose(
        (start_pose.position[0], start_pose.position[1], safe_height),
        start_pose.orientation,
    )
    waypoint2 = Pose(
        (block_pose.position[0], block_pose.position[1], safe_height),
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
        robot,
        end_effector_path,
        state.robot.joint_positions,
        collision_ids=set(),
        joint_distance_fn=joint_distance_fn,
        max_smoothing_iters_per_step=max_smoothing_iters_per_step,
    )

    plan: list[BlocksAction] = []
    plan.extend(_motion_plan_to_plan(motion_plan))
    plan.append(BlocksAction([0.0] * 7, gripper_action=-1))  # close
    return plan


def _get_place_block_plan(
    block_name: str,
    state: BlocksEnvState,
    robot: FingeredSingleArmPyBulletRobot,
    safe_height: float,
    block_height: float,
    joint_distance_fn: Callable[[JointPositions, JointPositions], float],
    max_smoothing_iters_per_step: int,
) -> list[BlocksAction]:
    # Reset the simulated robot to the given state.
    robot.set_joints(state.robot.joint_positions)

    start_pose = robot.get_end_effector_pose()
    block_pose = state.get_block_state(block_name).pose
    waypoint1 = Pose(
        (start_pose.position[0], start_pose.position[1], safe_height),
        start_pose.orientation,
    )
    waypoint2 = Pose(
        (block_pose.position[0], block_pose.position[1], safe_height),
        start_pose.orientation,
    )
    waypoint3 = Pose(
        (
            block_pose.position[0],
            block_pose.position[1],
            block_pose.position[2] + block_height,
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
        robot,
        end_effector_path,
        state.robot.joint_positions,
        collision_ids=set(),
        joint_distance_fn=joint_distance_fn,
        max_smoothing_iters_per_step=max_smoothing_iters_per_step,
    )

    plan = _motion_plan_to_plan(motion_plan)
    plan.append(BlocksAction([0.0] * 7, gripper_action=1))  # open
    return plan


class PickBlockSkill(BlocksSkill):
    """Pick up a block."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return PickOperator

    def _get_plan_given_objects(
        self, objects: Sequence[Object], obs: BlocksEnvState
    ) -> list[BlocksAction]:
        print(f"Getting plan for Pick({objects})")
        _, block, _ = objects
        return _get_pick_block_plan(
            block.name,
            obs,
            self._robot,
            self._safe_height,
            self._joint_distance_fn,
            self._max_smoothing_iters_per_step,
        )


class UnstackBlockSkill(BlocksSkill):
    """Unstack a block."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return UnstackOperator

    def _get_plan_given_objects(
        self, objects: Sequence[Object], obs: BlocksEnvState
    ) -> list[BlocksAction]:
        print(f"Getting plan for Unstack({objects})")
        _, block, _ = objects
        return _get_pick_block_plan(
            block.name,
            obs,
            self._robot,
            self._safe_height,
            self._joint_distance_fn,
            self._max_smoothing_iters_per_step,
        )


class StackBlockSkill(BlocksSkill):
    """Stack a block on another block."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return StackOperator

    def _get_plan_given_objects(
        self, objects: Sequence[Object], obs: BlocksEnvState
    ) -> list[BlocksAction]:
        print(f"Getting plan for Stack({objects})")
        _, _, block = objects
        block_height = 2 * self._scene_spec.block_half_extents[2]
        return _get_place_block_plan(
            block.name,
            obs,
            self._robot,
            self._safe_height,
            block_height,
            self._joint_distance_fn,
            self._max_smoothing_iters_per_step,
        )


class PlaceBlockOnTableSkill(BlocksSkill):
    """Place a block on the table."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return PlaceOperator

    def _get_plan_given_objects(
        self, objects: Sequence[Object], obs: BlocksEnvState
    ) -> list[BlocksAction]:
        print(f"Getting plan for Place({objects})")
        _, _, block = objects
        table_height = 2 * self._scene_spec.table_half_extents[2]
        return _get_place_block_plan(
            block.name,
            obs,
            self._robot,
            self._safe_height,
            table_height,
            self._joint_distance_fn,
            self._max_smoothing_iters_per_step,
        )


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

        # Create a simulator for kinematics etc.
        self._sim = BlocksEnv(scene_spec, seed=seed)

        # Create the perceiver.
        self._perceiver = BlocksPerceiver(self._sim)

        # Create the skills.
        skill_classes = {
            PickBlockSkill,
            PlaceBlockOnTableSkill,
            UnstackBlockSkill,
            StackBlockSkill,
        }
        self._skills: set[Skill[BlocksEnvState, BlocksAction]] = {
            c(
                seed,
                self._sim.robot,
                scene_spec,
                safe_height,
                max_smoothing_iters_per_step=max_smoothing_iters_per_step,
            )  # type: ignore
            for c in skill_classes
        }

        # Create the planner.
        self._planner = TaskThenMotionPlanner(
            TYPES,
            PREDICATES,
            self._perceiver,
            OPERATORS,
            self._skills,
            planner_id="pyperplan",
        )

        # Track the current goal.
        self._current_goal: BlocksCommand | None = None

    def reset(self, initial_state: BlocksEnvState) -> None:
        self._np_random = np.random.default_rng(self._seed)
        # NOTE: the planner is not yet reset because we don't have a goal until
        # a command is issued.
        self._current_goal = None

    def step_action_space(
        self, state: BlocksEnvState, command: BlocksCommand
    ) -> Space[BlocksAction]:
        return FunctionalSpace(
            contains_fn=lambda x: isinstance(x, BlocksAction),
            sample_fn=partial(self._get_action, state, command),
        )

    def _get_action(
        self,
        state: BlocksEnvState,
        command: BlocksCommand,
        rng: np.random.Generator,
    ) -> BlocksAction:
        del rng  # the randomization is buried right now
        if self._current_goal != command:
            # Replan.
            info = {"goal": command}
            self._planner.reset(state, info)
            self._current_goal = command
        return self._planner.step(state)

    def get_command_space(self) -> Space[BlocksCommand]:
        # TODO update to include actual towers
        return EnumSpace([BlocksCommand([])])
