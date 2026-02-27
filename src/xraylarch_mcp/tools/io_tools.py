"""I/O tools: load spectra, manage groups, read Athena projects."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from larch import Group
from mcp.server.fastmcp import FastMCP

from ..session import SessionManager
from ..util import format_error, summarize_group


def _get_session(ctx: Any) -> SessionManager:
    return ctx.request_context.lifespan_context["session"]


def register(mcp: FastMCP) -> None:
    @mcp.tool(name="larch_load_spectrum")
    def larch_load_spectrum(
        ctx: Any,
        filepath: str,
        format: str = "auto",
        group_id: str | None = None,
    ) -> dict:
        """Load an X-ray spectrum from a file.

        Supports XDI (.xdi), ASCII columnar (.dat, .txt), Athena projects (.prj),
        CSV (.csv), GSECARS (.gsexdi), Spec files, and more.
        Format is auto-detected from the file extension and header.

        Args:
            filepath: Path to the spectrum file.
            format: File format. Use "auto" for auto-detection, or specify
                    "xdi", "ascii", "athena", "csv", "gsexdi", "specfile".
            group_id: Optional custom ID for the loaded group. Auto-generated from
                      filename if not provided.

        Returns:
            Group ID, detected columns, energy range, and number of points.
        """
        session = _get_session(ctx)

        filepath = os.path.expanduser(filepath)
        if not os.path.isfile(filepath):
            return {"error": f"File not found: {filepath}"}

        try:
            ext = os.path.splitext(filepath)[1].lower()

            # Handle Athena project files specially
            if ext == ".prj" or format == "athena":
                from larch.io import read_athena

                project = read_athena(filepath)
                group_names = []
                for attr in dir(project):
                    if attr.startswith("_"):
                        continue
                    obj = getattr(project, attr)
                    if isinstance(obj, Group):
                        group_names.append(attr)

                if len(group_names) == 1:
                    # Single group - load it directly
                    grp = getattr(project, group_names[0])
                    gid = session.add_group(grp, group_id=group_id, filepath=filepath)
                    return {
                        "group_id": gid,
                        "source": "athena_project",
                        "arrays": list(
                            k
                            for k in dir(grp)
                            if not k.startswith("_")
                            and isinstance(getattr(grp, k, None), np.ndarray)
                        ),
                        "n_points": (
                            len(grp.energy) if hasattr(grp, "energy") else None
                        ),
                        "energy_range": (
                            [float(grp.energy.min()), float(grp.energy.max())]
                            if hasattr(grp, "energy")
                            else None
                        ),
                    }
                else:
                    # Multiple groups - list them
                    return {
                        "source": "athena_project",
                        "message": f"Athena project contains {len(group_names)} groups. Use larch_read_athena_project to inspect them, or load individually.",
                        "group_names": group_names,
                        "filepath": filepath,
                    }

            # Auto-detect or use specified format
            if format == "auto":
                from larch.io import guess_filereader

                reader_name = guess_filereader(filepath)
                if reader_name:
                    format = reader_name
                else:
                    format = "ascii"  # fallback

            # Map format to reader function
            if format in ("read_xdi", "xdi") or ext == ".xdi":
                from larch.io import read_xdi

                group = read_xdi(filepath)
            elif format in ("read_csv", "csv") or ext == ".csv":
                from larch.io import read_csv

                group = read_csv(filepath)
            elif format in ("read_gsexdi", "gsexdi"):
                from larch.io import read_gsexdi

                group = read_gsexdi(filepath)
            elif format in ("read_specfile", "specfile"):
                from larch.io import read_specfile

                group = read_specfile(filepath)
            else:
                from larch.io import read_ascii

                group = read_ascii(filepath)

            # Calculate mu from transmission columns if needed
            if not hasattr(group, "mu"):
                if hasattr(group, "i0") and hasattr(group, "it"):
                    i0 = np.array(group.i0, dtype=float)
                    it = np.array(group.it, dtype=float)
                    # Avoid log(0) or log(negative)
                    mask = (i0 > 0) & (it > 0)
                    mu = np.zeros_like(i0)
                    mu[mask] = np.log(i0[mask] / it[mask])
                    group.mu = mu

            gid = session.add_group(group, group_id=group_id, filepath=filepath)

            # Build response
            arrays = []
            for attr in sorted(dir(group)):
                if attr.startswith("_"):
                    continue
                val = getattr(group, attr, None)
                if isinstance(val, np.ndarray) and val.ndim == 1:
                    arrays.append(attr)

            energy_range = None
            n_points = None
            if hasattr(group, "energy"):
                energy_range = [
                    float(group.energy.min()),
                    float(group.energy.max()),
                ]
                n_points = len(group.energy)

            return {
                "group_id": gid,
                "format_used": format,
                "arrays": arrays,
                "n_points": n_points,
                "energy_range": energy_range,
                "has_mu": hasattr(group, "mu"),
            }

        except Exception as e:
            return {"error": format_error("larch_load_spectrum", e)}

    @mcp.tool(name="larch_read_athena_project")
    def larch_read_athena_project(
        ctx: Any,
        filepath: str,
        load_group: str | None = None,
        load_all: bool = False,
    ) -> dict:
        """Read an Athena project file (.prj) and list or load its groups.

        Args:
            filepath: Path to the .prj file.
            load_group: Name of a specific group to load from the project.
            load_all: If True, load all groups from the project.

        Returns:
            List of groups in the project, or loaded group info.
        """
        session = _get_session(ctx)
        filepath = os.path.expanduser(filepath)

        if not os.path.isfile(filepath):
            return {"error": f"File not found: {filepath}"}

        try:
            from larch.io import read_athena

            project = read_athena(filepath)
            group_names = []
            for attr in sorted(dir(project)):
                if attr.startswith("_"):
                    continue
                obj = getattr(project, attr)
                if isinstance(obj, Group):
                    group_names.append(attr)

            if load_group:
                if load_group not in group_names:
                    return {
                        "error": f"Group '{load_group}' not in project. Available: {group_names}"
                    }
                grp = getattr(project, load_group)
                gid = session.add_group(grp, label=load_group)
                return {
                    "loaded": True,
                    "group_id": gid,
                    "summary": summarize_group(grp, gid),
                }

            if load_all:
                loaded = []
                for name in group_names:
                    grp = getattr(project, name)
                    gid = session.add_group(grp, label=name)
                    loaded.append(gid)
                return {
                    "loaded": True,
                    "group_ids": loaded,
                    "count": len(loaded),
                }

            # List groups with basic info
            groups_info = []
            for name in group_names:
                grp = getattr(project, name)
                info: dict[str, Any] = {"name": name}
                if hasattr(grp, "energy"):
                    info["n_points"] = len(grp.energy)
                    info["energy_range"] = [
                        float(grp.energy.min()),
                        float(grp.energy.max()),
                    ]
                if hasattr(grp, "e0"):
                    info["e0"] = float(grp.e0)
                groups_info.append(info)

            return {
                "filepath": filepath,
                "n_groups": len(group_names),
                "groups": groups_info,
            }

        except Exception as e:
            return {"error": format_error("larch_read_athena_project", e)}

    @mcp.tool(name="larch_list_groups")
    def larch_list_groups(ctx: Any) -> dict:
        """List all loaded spectrum groups with summary info.

        Returns:
            List of all group IDs with arrays, processing state, key scalars.
        """
        session = _get_session(ctx)
        groups = session.list_groups()
        return {"groups": groups, "count": len(groups)}

    @mcp.tool(name="larch_inspect_group")
    def larch_inspect_group(ctx: Any, group_id: str) -> dict:
        """Inspect a loaded group's attributes in detail.

        Args:
            group_id: The ID of the group to inspect.

        Returns:
            Detailed listing of arrays (with shapes, min, max), scalars,
            and what processing has been applied.
        """
        session = _get_session(ctx)
        try:
            return session.inspect_group(group_id)
        except KeyError as e:
            return {"error": str(e)}

    @mcp.tool(name="larch_remove_group")
    def larch_remove_group(ctx: Any, group_id: str) -> dict:
        """Remove a loaded group to free memory.

        Args:
            group_id: The ID of the group to remove.

        Returns:
            Confirmation of removal.
        """
        session = _get_session(ctx)
        try:
            session.remove_group(group_id)
            return {"removed": group_id, "remaining": session.group_ids}
        except KeyError as e:
            return {"error": str(e)}
