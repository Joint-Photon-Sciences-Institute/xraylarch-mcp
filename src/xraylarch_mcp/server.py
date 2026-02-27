"""FastMCP server definition with tool registration and lifespan management."""

from __future__ import annotations

import io
import sys
import traceback
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from .session import SessionManager


@asynccontextmanager
async def larch_lifespan(app: FastMCP):
    """Initialize the larch session manager at server startup."""
    session = SessionManager()
    yield {"session": session}


mcp = FastMCP(
    "xraylarch-mcp",
    instructions=(
        "X-ray spectroscopy analysis server powered by xraylarch. "
        "Load XAS/XAFS spectra, normalize, extract EXAFS, perform Fourier transforms, "
        "fit data, and generate publication-quality plots. "
        "Start by loading a spectrum with larch_load_spectrum, then use the XAFS "
        "tools (larch_normalize, larch_autobk, larch_xftf) for analysis."
    ),
    lifespan=larch_lifespan,
)


def _get_session(ctx: Context) -> SessionManager:
    return ctx.request_context.lifespan_context["session"]


# Register the escape-hatch tool directly on the server
@mcp.tool(name="larch_run_code")
def larch_run_code(ctx: Context, code: str) -> dict:
    """Execute arbitrary Python code in the larch session context.

    The session manager, all loaded groups, numpy, and matplotlib are available.
    Use this for advanced operations not covered by the structured tools.

    Args:
        code: Python code to execute. Access groups via `groups` dict,
              numpy as `np`, matplotlib.pyplot as `plt`.

    Returns:
        stdout output and any result.
    """
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    from larch import Group
    from larch.xafs import autobk, pre_edge, xftf, xftr

    session = _get_session(ctx)

    # Build execution namespace
    namespace = {
        "np": np,
        "plt": plt,
        "Group": Group,
        "pre_edge": pre_edge,
        "autobk": autobk,
        "xftf": xftf,
        "xftr": xftr,
        "session": session,
        "groups": {gid: session.get_group(gid) for gid in session.group_ids},
    }

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = capture = io.StringIO()
    result_val = None

    try:
        # Try eval first (expression), fall back to exec (statements)
        try:
            result_val = eval(code, namespace)
        except SyntaxError:
            exec(code, namespace)
            result_val = namespace.get("result", None)

        stdout = capture.getvalue()
        response: dict[str, Any] = {"stdout": stdout}
        if result_val is not None:
            response["result"] = repr(result_val)
        return response

    except Exception as e:
        stdout = capture.getvalue()
        return {
            "stdout": stdout,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }
    finally:
        sys.stdout = old_stdout


# Register all tool modules
from .tools import fitting_tools, io_tools, math_tools, plot_tools, xafs_tools
from .resources import docs

io_tools.register(mcp)
xafs_tools.register(mcp)
plot_tools.register(mcp)
math_tools.register(mcp)
fitting_tools.register(mcp)
docs.register(mcp)
