"""Use a fresh workspace-owned pytest temporary directory on restricted Windows."""

from pathlib import Path
from uuid import uuid4

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Keep test outputs inside ignored workspace storage, never global temp."""
    if config.option.basetemp is None:
        parent = Path(__file__).resolve().parents[1] / ".venv" / "pytest_runs"
        parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(parent / uuid4().hex)
