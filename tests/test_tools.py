"""Tests that call the registered MCP tools, on larch's example data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from larch import Group
from larch.io import read_athena, read_xdi

from tests.conftest import MockContext
from xraylarch_mcp.server import mcp
from xraylarch_mcp.session import SessionManager

DATA = Path(__file__).parent / "data"
CU = str(DATA / "cu_metal_rt.xdi")
FEFF = str(DATA / "feffcu01.dat")
PRJ = str(DATA / "fe_athena.prj")
FLUO = str(DATA / "fe_fluorescence.xdi")


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


@pytest.fixture
def fe_ctx() -> MockContext:
    """The FeO, Fe2O3 and Fe3C references of fe_athena.prj, as 'feo', 'fe2o3', 'fe3c'."""
    session = SessionManager()
    project = read_athena(PRJ)
    for gid, name in (("feo", "feo_rt1_xmu"), ("fe2o3", "fe2o3_rt1_xmu"), ("fe3c", "fe3c_rt_xdi")):
        session.add_group(getattr(project, name), group_id=gid)
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


def test_lcf_reports_weights_by_group_id(fe_ctx):
    """A 0.6 FeO + 0.4 Fe2O3 mixture of the normalized references."""
    session = fe_ctx.session
    for gid in ("feo", "fe2o3"):
        assert "error" not in call(fe_ctx, "larch_normalize", group_id=gid)
    feo, fe2o3 = session.get_group("feo"), session.get_group("fe2o3")
    mixture = Group(
        energy=feo.energy.copy(),
        norm=0.6 * feo.norm + 0.4 * np.interp(feo.energy, fe2o3.energy, fe2o3.norm),
    )
    session.add_group(mixture, group_id="mix")
    fit = call(fe_ctx, "larch_lcf", group_id="mix", standard_group_ids=["feo", "fe2o3"],
               xmin=7100.0, xmax=7200.0)
    assert "error" not in fit, fit
    assert fit["weights"]["feo"]["value"] == pytest.approx(0.6, abs=0.01)
    assert fit["weights"]["fe2o3"]["value"] == pytest.approx(0.4, abs=0.01)
    assert fit["weights"]["feo"]["stderr"] is not None
    assert fit["e0_shift"] is None


def test_mback_norm_runs_and_keeps_the_pre_edge_step(fe_ctx):
    assert "error" not in call(fe_ctx, "larch_normalize", group_id="feo")
    out = call(fe_ctx, "larch_mback_norm", group_id="feo", z=26)
    assert "error" not in out, out
    assert out["edge_step"] == pytest.approx(1.321, abs=0.005)
    assert out["edge_step_poly"] == pytest.approx(1.354, abs=0.005)


def test_fluo_corr_puts_the_corrected_spectrum_in_a_new_group():
    session = SessionManager()
    group = read_xdi(FLUO)
    group.mu = group.fe_ka / group.i0
    session.add_group(group, group_id="fe")
    ctx = MockContext(session)
    assert "error" not in call(ctx, "larch_normalize", group_id="fe")
    out = call(ctx, "larch_fluo_corr", group_id="fe", formula="Fe2O3", elem="Fe")
    assert "error" not in out, out
    assert out["group_id"] == "fe_fluo_corr" and out["source_group_id"] == "fe"
    assert out["norm_max_before"] == pytest.approx(1.035, abs=0.002)
    assert out["norm_max_corrected"] == pytest.approx(1.171, abs=0.002)
    np.testing.assert_allclose(session.get_group("fe_fluo_corr").mu, group.mu_corr)
