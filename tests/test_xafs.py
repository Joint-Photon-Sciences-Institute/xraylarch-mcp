"""Tests for XAFS analysis tools."""

from __future__ import annotations

import numpy as np
import pytest
from larch import Group
from larch.xafs import autobk, pre_edge, xftf, xftr

from tests.conftest import MockContext, make_synthetic_xas
from xraylarch_mcp.session import SessionManager


class TestNormalization:
    def test_pre_edge_runs(self, synthetic_group: Group):
        """pre_edge should add norm, e0, edge_step to the group."""
        pre_edge(synthetic_group)
        assert hasattr(synthetic_group, "norm")
        assert hasattr(synthetic_group, "e0")
        assert hasattr(synthetic_group, "edge_step")

    def test_e0_detection(self, synthetic_group: Group):
        """E0 should be detected near the synthetic edge at 7112 eV."""
        pre_edge(synthetic_group)
        assert abs(synthetic_group.e0 - 7112.0) < 5.0

    def test_edge_step_positive(self, synthetic_group: Group):
        """Edge step should be positive for a normal absorption edge."""
        pre_edge(synthetic_group)
        assert synthetic_group.edge_step > 0

    def test_norm_shape(self, synthetic_group: Group):
        """Normalized spectrum should have same shape as input."""
        pre_edge(synthetic_group)
        assert synthetic_group.norm.shape == synthetic_group.energy.shape

    def test_norm_near_one_above_edge(self, synthetic_group: Group):
        """Normalized spectrum should be near 1 well above edge."""
        pre_edge(synthetic_group)
        # Get norm values well above the edge (e0 + 100 to e0 + 200)
        mask = (synthetic_group.energy > synthetic_group.e0 + 100) & (
            synthetic_group.energy < synthetic_group.e0 + 200
        )
        norm_above = synthetic_group.norm[mask]
        assert np.mean(norm_above) == pytest.approx(1.0, abs=0.15)


class TestAutobk:
    def test_autobk_runs(self, synthetic_group: Group):
        """autobk should produce chi(k) and k arrays."""
        pre_edge(synthetic_group)
        autobk(synthetic_group, rbkg=1.0)
        assert hasattr(synthetic_group, "k")
        assert hasattr(synthetic_group, "chi")
        assert hasattr(synthetic_group, "bkg")

    def test_k_positive(self, synthetic_group: Group):
        """k values should be non-negative."""
        pre_edge(synthetic_group)
        autobk(synthetic_group, rbkg=1.0)
        assert synthetic_group.k.min() >= 0

    def test_chi_shape(self, synthetic_group: Group):
        """chi and k should have the same shape."""
        pre_edge(synthetic_group)
        autobk(synthetic_group, rbkg=1.0)
        assert synthetic_group.chi.shape == synthetic_group.k.shape


class TestFourierTransform:
    @pytest.fixture
    def processed_group(self, synthetic_group: Group) -> Group:
        pre_edge(synthetic_group)
        autobk(synthetic_group, rbkg=1.0)
        return synthetic_group

    def test_xftf_runs(self, processed_group: Group):
        """xftf should produce R, chir_mag arrays."""
        xftf(processed_group, kmin=2, kmax=6, dk=2, window="kaiser", kweight=2)
        assert hasattr(processed_group, "r")
        assert hasattr(processed_group, "chir_mag")

    def test_r_positive(self, processed_group: Group):
        """R values should be non-negative."""
        xftf(processed_group, kmin=2, kmax=6, dk=2, window="kaiser", kweight=2)
        assert processed_group.r.min() >= 0

    def test_xftr_runs(self, processed_group: Group):
        """xftr should produce q, chiq_mag arrays."""
        xftf(processed_group, kmin=2, kmax=6, dk=2, window="kaiser", kweight=2)
        xftr(processed_group, rmin=0.5, rmax=3.0, dr=0.1)
        assert hasattr(processed_group, "q")
        assert hasattr(processed_group, "chiq_mag")


class TestToolWrappers:
    """Test the MCP tool wrapper functions using mock context."""

    def test_normalize_tool(self, loaded_ctx):
        ctx, gid = loaded_ctx
        session = ctx.session

        from xraylarch_mcp.server import mcp
        # Get internal function
        from larch.xafs import pre_edge
        group = session.get_group(gid)
        pre_edge(group)
        assert hasattr(group, "norm")
        assert hasattr(group, "e0")

    def test_autobk_tool(self, loaded_ctx):
        ctx, gid = loaded_ctx
        session = ctx.session
        group = session.get_group(gid)

        from larch.xafs import autobk, pre_edge
        pre_edge(group)
        autobk(group, rbkg=1.0)
        assert hasattr(group, "k")
        assert hasattr(group, "chi")

    def test_full_pipeline(self, loaded_ctx):
        """Test the full XANES -> EXAFS -> FT pipeline."""
        ctx, gid = loaded_ctx
        session = ctx.session
        group = session.get_group(gid)

        from larch.xafs import autobk, pre_edge, xftf

        pre_edge(group)
        assert abs(group.e0 - 7112.0) < 5.0

        autobk(group, rbkg=1.0)
        assert len(group.k) > 10

        xftf(group, kmin=2, kmax=float(group.k.max()), dk=2, window="kaiser", kweight=2)
        assert len(group.r) > 10
        assert group.chir_mag.max() > 0
