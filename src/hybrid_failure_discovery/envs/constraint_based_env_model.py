"""A constraint-based model of an environment."""

import abc
from typing import Any, Generic

from gymnasium.core import ActType, Env, ObsType, RenderFrame
from gymnasium.spaces import Space
from tomsutils.utils import sample_seed_from_rng


class ConstraintBasedEnvModel(Generic[ObsType, ActType]):
    """A constraint-based model of an environment."""

    @abc.abstractmethod
    def get_description(self) -> str:
        """Get a human-readable description of the environment."""

    @abc.abstractmethod
    def get_initial_states(self) -> Space[ObsType]:
        """Get the set of all possible initial states."""

    @abc.abstractmethod
    def get_next_states(self, state: ObsType, action: ActType) -> Space[ObsType]:
        """Get the set of all possible next states (transition constraint)."""

    @abc.abstractmethod
    def actions_are_equal(self, action1: ActType, action2: ActType) -> bool:
        """Check if two actions are equivalent in this environment."""


class ConstraintBasedGymEnv(
    Env[ObsType, ActType], ConstraintBasedEnvModel[ObsType, ActType]
):
    """An OpenAI Gym environment defined by a constraint based env model."""

    def __init__(self, seed: int = 0) -> None:
        self._np_random_seed = seed
        self.action_space = self._create_action_space()
        self._current_state: ObsType | None = None  # set in reset()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        initial_states = self.get_initial_states()
        assert self._np_random is not None
        initial_states.seed(sample_seed_from_rng(self._np_random))
        self._current_state = initial_states.sample()
        return self._get_obs(), self._get_info()

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        assert self._current_state is not None
        next_states = self.get_next_states(self._current_state, action)
        assert self._np_random is not None
        next_states.seed(sample_seed_from_rng(self._np_random))
        next_state = next_states.sample()
        reward, terminated = self._get_reward_and_termination(
            self._current_state, action, next_state
        )
        self._current_state = next_state
        truncated = False
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        assert self._current_state is not None
        return self._render_state(self._current_state)

    @abc.abstractmethod
    def _create_action_space(self) -> Space[ActType]:
        """Create the action space."""

    @abc.abstractmethod
    def _get_obs(self) -> ObsType:
        """Get the current observation."""

    @abc.abstractmethod
    def _get_reward_and_termination(
        self, state: ObsType, action: ActType, next_state: ObsType
    ) -> tuple[float, bool]:
        """Get reward and termination."""

    @abc.abstractmethod
    def _render_state(self, state: ObsType) -> RenderFrame | list[RenderFrame] | None:
        """Render the given state."""

    def _get_info(self) -> dict[str, Any]:
        return {}
