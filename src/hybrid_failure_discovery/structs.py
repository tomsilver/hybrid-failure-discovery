"""Data structures."""

from dataclasses import dataclass
from typing import Callable, Generic, TypeAlias, TypeVar

from gymnasium.core import ActType, ObsType

CommandType = TypeVar("CommandType")


@dataclass
class Trajectory(Generic[ObsType, ActType, CommandType]):
    """A trajectory is a sequence of observations and actions."""

    observations: list[ObsType]
    actions: list[ActType]
    commands: list[CommandType]


TrajectoryHeuristic: TypeAlias = Callable[[Trajectory], float]
