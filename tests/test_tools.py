"""Tests that call the registered MCP tools, on larch's example data."""

from __future__ import annotations

from pathlib import Path

import pytest
from larch.io import read_athena, read_xdi

from tests.conftest import MockContext
from xraylarch_mcp.server import mcp
from xraylarch_mcp.session import SessionManager

DATA = Path(__file__).parent / "data"
CU = str(DATA / "cu_metal_rt.xdi")
FEFF = str(DATA / "feffcu01.dat")
PRJ = str(DATA / "fe_athena.prj")


def call(ctx: MockContext, name: str, **arguments) -> dict:
    return mcp._tool_manager.get_tool(name).fn(ctx=ctx, **arguments)


@pytest.fixture
def cu_ctx() -> MockContext:
    """Cu foil at room temperature, as group 'cu'."""
    session = SessionManager()
    group = read_xdi(CU)
    group.mu = group.mutrans
    session.add_group(group, group_id="cu")
    return MockContext(session)


def test_feffit_varies_parameters_and_matches_larch_example(cu_ctx):
    """Cu first shell, as in larch's examples/feffit/doc_feffit1.py."""
    assert "error" not in call(cu_ctx, "larch_normalize", group_id="cu")
    assert "error" not in call(cu_ctx, "larch_autobk", group_id="cu", rbkg=1.0, kweight=2)
    guesses = {"s02": "guess:1.0", "sigma2": "guess:0.0", "deltar": "guess:0.0", "e0": "guess:0.0"}
    fit = call(
        cu_ctx, "larch_feffit", group_id="cu", paths=[{"filename": FEFF, **guesses}],
        kmin=3, kmax=17, kweight=2, dk=4, window="kaiser", rmin=1.4, rmax=3.0, fitspace="r",
    )
    assert "error" not in fit, fit
    params = {name: p["value"] for name, p in fit["parameters"].items()}
    assert params["s02_0"] == pytest.approx(0.931, abs=0.08)
    assert params["sigma2_0"] == pytest.approx(0.00867, abs=0.0008)
    assert params["deltar_0"] == pytest.approx(-0.0060, abs=0.006)
    assert params["e0_0"] == pytest.approx(4.36, abs=1.5)
    assert all(p["stderr"] is not None for p in fit["parameters"].values())
    assert fit["rfactor"] < 0.02
    assert fit["chi_reduced"] == pytest.approx(fit["chi_square"] / (fit["n_independent"] - 4), rel=1e-3)
