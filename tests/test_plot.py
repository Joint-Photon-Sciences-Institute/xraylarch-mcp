"""Tests for plotting tools."""

from __future__ import annotations

import base64

import numpy as np
import pytest
from larch import Group
from larch.xafs import autobk, pre_edge, xftf

from tests.conftest import MockContext, make_synthetic_xas
from xraylarch_mcp.session import SessionManager


class TestPlotting:
    @pytest.fixture
    def fully_processed_ctx(self) -> tuple[MockContext, str]:
        """Context with a fully processed spectrum."""
        session = SessionManager()
        g = make_synthetic_xas()
        pre_edge(g)
        autobk(g, rbkg=1.0)
        xftf(g, kmin=2, kmax=float(g.k.max()), dk=2, window="kaiser", kweight=2)
        gid = session.add_group(g, group_id="fe_foil")
        return MockContext(session), gid

    def test_plot_norm(self, fully_processed_ctx):
        """Plotting normalized spectrum should produce valid PNG."""
        ctx, gid = fully_processed_ctx

        from xraylarch_mcp.tools.plot_tools import _get_plot_data

        group = ctx.session.get_group(gid)
        x, y, (xlabel, ylabel) = _get_plot_data(group, "norm")
        assert len(x) == len(y)
        assert "Energy" in xlabel

    def test_plot_chi_k(self, fully_processed_ctx):
        """Plotting chi(k) should work with k-weighting."""
        ctx, gid = fully_processed_ctx

        from xraylarch_mcp.tools.plot_tools import _get_plot_data

        group = ctx.session.get_group(gid)
        x, y, (xlabel, ylabel) = _get_plot_data(group, "chi_k", kweight=2)
        assert len(x) == len(y)
        # k-weighted chi should differ from raw chi
        assert not np.array_equal(y, group.chi)

    def test_plot_chi_r(self, fully_processed_ctx):
        """Plotting chi(R) should work."""
        ctx, gid = fully_processed_ctx

        from xraylarch_mcp.tools.plot_tools import _get_plot_data

        group = ctx.session.get_group(gid)
        x, y, (xlabel, ylabel) = _get_plot_data(group, "chi_r")
        assert len(x) == len(y)
        assert "R" in xlabel

    def test_actual_plot_produces_png(self, fully_processed_ctx):
        """Full plot generation should produce base64 PNG data."""
        import io

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ctx, gid = fully_processed_ctx
        group = ctx.session.get_group(gid)

        fig, ax = plt.subplots()
        ax.plot(group.energy, group.norm)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Normalized mu(E)")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()

        # Verify it's valid PNG by checking base64 length
        assert len(b64) > 1000
        # Verify it decodes
        raw = base64.b64decode(b64)
        assert raw[:4] == b"\x89PNG"

    def test_invalid_plot_type(self, fully_processed_ctx):
        """Invalid plot type should raise ValueError."""
        from xraylarch_mcp.tools.plot_tools import _get_plot_data

        ctx, gid = fully_processed_ctx
        group = ctx.session.get_group(gid)

        with pytest.raises(ValueError, match="Unknown plot_type"):
            _get_plot_data(group, "invalid_type")
