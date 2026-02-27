"""Interactive tools that generate browser-based HTML interfaces."""

from __future__ import annotations

import json
import tempfile
import webbrowser
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
        html_path: str | None = None,
    ) -> dict:
        """Open an interactive HTML page to select normalization parameters.

        Generates a self-contained HTML file with Plotly.js plots and sliders
        for adjusting pre-edge and post-edge normalization parameters in real
        time. The page opens automatically in the default web browser.

        Workflow:
            1. This tool generates and opens the HTML page.
            2. The user adjusts sliders until the normalization looks right.
            3. The user clicks "Save Parameters" to download a JSON file,
               or copies the parameters from the display.
            4. Use larch_apply_norm_params to apply the saved parameters.

        Args:
            group_id: ID of the loaded spectrum group.
            e0: Initial edge energy in eV. Auto-detected if omitted.
            pre1: Initial pre-edge lower bound relative to E0.
            pre2: Initial pre-edge upper bound relative to E0.
            norm1: Initial post-edge lower bound relative to E0.
            norm2: Initial post-edge upper bound relative to E0.
            nnorm: Initial post-edge polynomial degree (1-3).
            html_path: Optional path to write the HTML file. If omitted,
                       a temporary file is created.

        Returns:
            Path to the HTML file and instructions for next steps.
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

        try:
            from larch.xafs import pre_edge

            from ..interactive.norm_html import generate_norm_html

            # Run pre_edge to get auto-detected defaults
            pre_edge(group)

            # Use auto-detected values where user didn't specify
            details = group.pre_edge_details
            e0_val = e0 if e0 is not None else float(group.e0)
            pre1_val = pre1 if pre1 is not None else float(details.pre1)
            pre2_val = pre2 if pre2 is not None else float(details.pre2)
            norm1_val = norm1 if norm1 is not None else float(details.norm1)
            norm2_val = norm2 if norm2 is not None else float(details.norm2)
            nnorm_val = nnorm if nnorm is not None else int(details.nnorm)
            edge_step_val = float(group.edge_step)

            label = getattr(group, "label", None) or group_id

            # Generate the HTML
            html_content = generate_norm_html(
                energy=group.energy,
                mu=group.mu,
                e0=e0_val,
                pre1=pre1_val,
                pre2=pre2_val,
                norm1=norm1_val,
                norm2=norm2_val,
                nnorm=nnorm_val,
                edge_step=edge_step_val,
                label=label,
                group_id=group_id,
            )

            # Write the HTML file
            if html_path:
                out_path = Path(html_path)
            else:
                tmp = tempfile.NamedTemporaryFile(
                    prefix=f"norm_{group_id}_",
                    suffix=".html",
                    delete=False,
                    mode="w",
                    encoding="utf-8",
                )
                out_path = Path(tmp.name)
                tmp.write(html_content)
                tmp.close()

            if html_path:
                out_path.write_text(html_content, encoding="utf-8")

            # Open in default browser (non-blocking)
            webbrowser.open(out_path.as_uri())

            return {
                "html_path": str(out_path),
                "group_id": group_id,
                "initial_params": {
                    "e0": e0_val,
                    "pre1": pre1_val,
                    "pre2": pre2_val,
                    "norm1": norm1_val,
                    "norm2": norm2_val,
                    "nnorm": nnorm_val,
                    "edge_step": edge_step_val,
                },
                "instructions": (
                    "Interactive normalization page opened in your browser. "
                    "Adjust the sliders, then click 'Save Parameters' to "
                    "download a JSON file. Tell me when you're done and I'll "
                    "apply the parameters with larch_apply_norm_params."
                ),
            }

        except Exception as e:
            return {"error": format_error("larch_interactive_norm", e)}

    @mcp.tool(name="larch_apply_norm_params")
    def larch_apply_norm_params(
        ctx: Context,
        group_id: str,
        params_path: str | None = None,
        e0: float | None = None,
        pre1: float | None = None,
        pre2: float | None = None,
        norm1: float | None = None,
        norm2: float | None = None,
        nnorm: int | None = None,
    ) -> dict:
        """Apply normalization parameters to a group.

        Read parameters from a JSON file (saved from the interactive HTML tool)
        or accept them directly as arguments, and apply via pre_edge().

        Args:
            group_id: ID of the spectrum group to normalize.
            params_path: Path to a JSON parameters file (from the interactive
                         normalization tool). If provided, file values are used
                         as defaults, and any explicit arguments override them.
            e0: Edge energy in eV.
            pre1: Pre-edge lower bound relative to E0.
            pre2: Pre-edge upper bound relative to E0.
            norm1: Post-edge lower bound relative to E0.
            norm2: Post-edge upper bound relative to E0.
            nnorm: Post-edge polynomial degree (1-3).

        Returns:
            Applied parameters and resulting edge_step.
        """
        session = _get_session(ctx)
        try:
            group = session.get_group(group_id)
        except KeyError as e:
            return {"error": str(e)}

        try:
            # Start from file params if provided
            file_params: dict[str, Any] = {}
            if params_path:
                p = Path(params_path)
                if not p.exists():
                    return {"error": f"Parameters file not found: {params_path}"}
                file_params = json.loads(p.read_text(encoding="utf-8"))

            # Build kwargs: file values as base, explicit args override
            kwargs: dict[str, Any] = {}
            for key, arg_val in [
                ("e0", e0),
                ("pre1", pre1),
                ("pre2", pre2),
                ("norm1", norm1),
                ("norm2", norm2),
                ("nnorm", nnorm),
            ]:
                if arg_val is not None:
                    kwargs[key] = arg_val
                elif key in file_params and file_params[key] is not None:
                    kwargs[key] = file_params[key]

            from larch.xafs import pre_edge

            pre_edge(group, **kwargs)

            result: dict[str, Any] = {
                "group_id": group_id,
                "e0": float(group.e0),
                "edge_step": float(group.edge_step),
                "applied_params": kwargs,
            }

            if hasattr(group, "pre_edge_details"):
                details = group.pre_edge_details
                result["pre_edge_details"] = {
                    "pre1": getattr(details, "pre1", None),
                    "pre2": getattr(details, "pre2", None),
                    "norm1": getattr(details, "norm1", None),
                    "norm2": getattr(details, "norm2", None),
                    "nnorm": getattr(details, "nnorm", None),
                }

            return result

        except Exception as e:
            return {"error": format_error("larch_apply_norm_params", e)}
