"""Shared pytest configuration and fixtures."""

from pathlib import Path
from typing import Any

import gymnasium as gym
import pytest
from gymnasium.wrappers import RecordVideo


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --make-videos CLI flag."""
    parser.addoption(
        "--make-videos",
        action="store_true",
        default=False,
        help="Save videos for tests marked with @pytest.mark.make_videos.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the make_videos marker."""
    config.addinivalue_line(
        "markers",
        "make_videos: save a video when --make-videos is passed.",
    )


@pytest.fixture()
def maybe_record(
    request: pytest.FixtureRequest,
) -> Any:
    """Wrap an env with RecordVideo if --make-videos is passed.

    Usage in tests::

        @pytest.mark.make_videos
        def test_something(maybe_record):
            env = maybe_record(MyEnv())
            ...
    """
    make_videos = request.config.getoption("--make-videos")

    def _wrap(env: gym.Env) -> gym.Env:  # type: ignore[type-arg]
        if not make_videos:
            return env
        videos_dir = Path(__file__).parent.parent / "videos"
        videos_dir.mkdir(exist_ok=True)
        test_name = request.node.name
        return RecordVideo(
            env,
            video_folder=str(videos_dir),
            name_prefix=test_name,
            episode_trigger=lambda _: True,
        )

    return _wrap
