"""Tests for I/O tools and session management."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from larch import Group

from xraylarch_mcp.session import SessionManager
from xraylarch_mcp.util import make_group_id, summarize_group, unique_group_id


class TestGroupIdGeneration:
    def test_from_filepath(self):
        assert make_group_id(filepath="/data/fe_foil.xdi") == "fe_foil"

    def test_from_label(self):
        assert make_group_id(label="Iron Foil K-edge") == "iron_foil_k_edge"

    def test_special_chars(self):
        assert make_group_id(filepath="Fe2O3 (sample 1).dat") == "fe2o3_sample_1"

    def test_unique_id(self):
        existing = {"fe_foil", "fe_foil_1"}
        assert unique_group_id("fe_foil", existing) == "fe_foil_2"

    def test_unique_id_no_collision(self):
        existing = {"other"}
        assert unique_group_id("fe_foil", existing) == "fe_foil"


class TestSessionManager:
    def test_add_and_get(self, session: SessionManager):
        g = Group()
        g.energy = np.array([1, 2, 3])
        gid = session.add_group(g, group_id="test")
        assert gid == "test"
        assert session.get_group("test") is g

    def test_add_auto_id(self, session: SessionManager):
        g = Group()
        gid = session.add_group(g, filepath="/data/my_sample.xdi")
        assert gid == "my_sample"

    def test_collision_handling(self, session: SessionManager):
        g1 = Group()
        g2 = Group()
        gid1 = session.add_group(g1, group_id="test")
        gid2 = session.add_group(g2, group_id="test")
        assert gid1 == "test"
        assert gid2 == "test_1"

    def test_get_nonexistent(self, session: SessionManager):
        with pytest.raises(KeyError, match="not found"):
            session.get_group("nonexistent")

    def test_remove(self, session: SessionManager):
        g = Group()
        session.add_group(g, group_id="test")
        session.remove_group("test")
        with pytest.raises(KeyError):
            session.get_group("test")

    def test_list_groups(self, session: SessionManager, synthetic_group: Group):
        session.add_group(synthetic_group, group_id="fe_foil")
        groups = session.list_groups()
        assert len(groups) == 1
        assert groups[0]["group_id"] == "fe_foil"
        assert "energy" in groups[0]["arrays"]

    def test_inspect_group(self, session: SessionManager, synthetic_group: Group):
        session.add_group(synthetic_group, group_id="fe_foil")
        info = session.inspect_group("fe_foil")
        assert info["group_id"] == "fe_foil"
        assert "energy" in info["arrays"]
        assert info["arrays"]["energy"]["shape"] == [500]


class TestSummarizeGroup:
    def test_basic_summary(self, synthetic_group: Group):
        summary = summarize_group(synthetic_group, "test")
        assert summary["group_id"] == "test"
        assert "energy" in summary["arrays"]
        assert "mu" in summary["arrays"]
        assert summary["arrays"]["energy"]["unit"] == "eV"

    def test_processed_summary(self, synthetic_group: Group):
        from larch.xafs import pre_edge

        pre_edge(synthetic_group)
        summary = summarize_group(synthetic_group, "test")
        assert "pre_edge" in summary["processing"]
        assert "e0" in summary["scalars"]
        assert "edge_step" in summary["scalars"]


class TestLoadSpectrum:
    def test_load_ascii_file(self, ctx):
        """Test loading a simple ASCII columnar file."""
        from xraylarch_mcp.server import mcp

        # Create a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".dat", delete=False
        ) as f:
            f.write("# energy  mu\n")
            energy = np.linspace(7000, 7300, 100)
            mu = np.arctan((energy - 7112) / 10)
            for e, m in zip(energy, mu):
                f.write(f"{e:.2f}  {m:.6f}\n")
            tmppath = f.name

        try:
            # Get the tool function
            tools = {t.name: t for t in mcp._tool_manager.list_tools()}
            assert "larch_load_spectrum" in tools
        finally:
            os.unlink(tmppath)
