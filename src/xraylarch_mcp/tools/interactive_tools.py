"""Interactive GUI tools that launch as subprocesses with a display backend."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from mcp.server.fastmcp import Context, FastMCP

from ..session import SessionManager
from ..util import format_error


def _get_session(ctx: Context) -> SessionManager:
    return ctx.request_context.lifespan_context["session"]


def register(mcp: FastMCP) -> None:
    @mcp.tool(name="larch_interactive_norm")
    def larch_interactive_norm(
        ctx: Context,
        group_id: str,
        e0: float | None = None,
        pre1: float | None = None,
        pre2: float | None = None,
        norm1: float | None = None,
        norm2: float | None = None,
        nnorm: int | None = None,
        save_path: str | None = None,
    ) -> dict:
        """Open an interactive GUI to select normalization parameters.

        Launches a matplotlib window with sliders for pre-edge and post-edge
        normalization parameters. Adjust the sliders to see the effect in
        real time, then close the window (or click Save) to apply.

        The chosen parameters are applied to the group via pre_edge().

        Args:
            group_id: ID of the loaded spectrum group.
            e0: Initial edge energy in eV. Auto-detected if omitted.
            pre1: Initial pre-edge lower bound relative to E0.
            pre2: Initial pre-edge upper bound relative to E0.
            norm1: Initial post-edge lower bound relative to E0.
            norm2: Initial post-edge upper bound relative to E0.
            nnorm: Initial post-edge polynomial degree (1-3).
            save_path: Optional path to write a plain-text parameters file.

        Returns:
            The selected normalization parameters dict.
        """
        session = _get_session(ctx)
        try:
            group = session.get_group(group_id)
        except KeyError as e:
            return {"error": str(e)}

        if not hasattr(group, "energy") or not hasattr(group, "mu"):
            return {
                "error": f"Group '{group_id}' missing 'energy' and/or 'mu' arrays. "
                f"Load a spectrum first."
            }

        # Write data to a temp npz file
        tmp_dir = tempfile.mkdtemp(prefix="larch_inorm_")
        npz_path = str(Path(tmp_dir) / "data.npz")
        json_path = str(Path(tmp_dir) / "params.json")

        try:
            np.savez(npz_path, energy=group.energy, mu=group.mu)

            # Build subprocess command
            cmd = [
                sys.executable, "-m", "xraylarch_mcp.interactive.norm",
                npz_path, json_path,
            ]
            if e0 is not None:
                cmd += ["--e0", str(e0)]
            if pre1 is not None:
                cmd += ["--pre1", str(pre1)]
            if pre2 is not None:
                cmd += ["--pre2", str(pre2)]
            if norm1 is not None:
                cmd += ["--norm1", str(norm1)]
            if norm2 is not None:
                cmd += ["--norm2", str(norm2)]
            if nnorm is not None:
                cmd += ["--nnorm", str(nnorm)]

            label = getattr(group, "label", None) or group_id
            cmd += ["--label", str(label)]

            # Launch subprocess (blocks until window is closed)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10-minute timeout
            )

            if result.returncode != 0:
                return {
                    "error": f"Interactive normalization subprocess failed "
                    f"(exit code {result.returncode})",
                    "stderr": result.stderr,
                    "stdout": result.stdout,
                }

            # Read back the chosen parameters
            json_file = Path(json_path)
            if not json_file.exists():
                return {
                    "error": "No parameters file was written by the interactive tool.",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }

            params = json.loads(json_file.read_text())

            # Apply the chosen parameters to the group
            from larch.xafs import pre_edge

            kwargs: dict[str, Any] = {}
            for key in ("e0", "pre1", "pre2", "norm1", "norm2", "nnorm"):
                if key in params and params[key] is not None:
                    kwargs[key] = params[key]

            pre_edge(group, **kwargs)

            # Optionally write a plain-text params file
            if save_path:
                with open(save_path, "w") as f:
                    f.write("# Normalization parameters (interactive)\n")
                    for key in ("e0", "pre1", "pre2", "norm1", "norm2", "nnorm"):
                        f.write(f"{key:<10s} = {params.get(key)}\n")
                    f.write(f"{'edge_step':<10s} = {float(group.edge_step)}\n")
                params["save_path"] = save_path

            params["group_id"] = group_id
            params["edge_step"] = float(group.edge_step)
            return params

        except subprocess.TimeoutExpired:
            return {
                "error": "Interactive normalization window timed out after 10 minutes."
            }
        except Exception as e:
            return {"error": format_error("larch_interactive_norm", e)}
        finally:
            # Clean up temp files
            for p in (npz_path, json_path):
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    pass
            try:
                Path(tmp_dir).rmdir()
            except OSError:
                pass
