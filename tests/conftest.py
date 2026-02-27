"""Shared test fixtures: synthetic spectra, session manager, mock context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from larch import Group

from xraylarch_mcp.session import SessionManager


def make_synthetic_xas(e0: float = 7112.0, n_points: int = 500) -> Group:
    """Create a synthetic Fe K-edge XAS spectrum using an arctangent edge."""
    g = Group()
    g.energy = np.linspace(e0 - 150, e0 + 300, n_points)
    g.mu = np.arctan((g.energy - e0) / 10)
    return g


@dataclass
class MockLifespanContext:
    session: SessionManager = field(default_factory=SessionManager)


@dataclass
class MockRequestContext:
    lifespan_context: dict = field(default_factory=dict)


class MockContext:
    """Mock MCP context that provides the session via lifespan_context."""

    def __init__(self, session: SessionManager | None = None):
        self._session = session or SessionManager()
        self.request_context = MockRequestContext(
            lifespan_context={"session": self._session}
        )

    @property
    def session(self) -> SessionManager:
        return self._session


@pytest.fixture
def session() -> SessionManager:
    return SessionManager()


@pytest.fixture
def ctx(session: SessionManager) -> MockContext:
    return MockContext(session)


@pytest.fixture
def synthetic_group() -> Group:
    return make_synthetic_xas()


@pytest.fixture
def loaded_session(session: SessionManager, synthetic_group: Group) -> tuple[SessionManager, str]:
    """Session with a pre-loaded synthetic spectrum."""
    gid = session.add_group(synthetic_group, group_id="fe_foil")
    return session, gid


@pytest.fixture
def loaded_ctx(loaded_session: tuple[SessionManager, str]) -> tuple[MockContext, str]:
    """Mock context with a pre-loaded synthetic spectrum."""
    session, gid = loaded_session
    return MockContext(session), gid
