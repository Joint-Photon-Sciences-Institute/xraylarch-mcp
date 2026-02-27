"""Math utility tools: smoothing, derivatives, interpolation, merging."""

from __future__ import annotations

from typing import Any

import numpy as np
from larch import Group
from mcp.server.fastmcp import Context, FastMCP

from ..session import SessionManager
from ..util import format_error


def _get_session(ctx: Context) -> SessionManager:
    return ctx.request_context.lifespan_context["session"]


def register(mcp: FastMCP) -> None:
    @mcp.tool(name="larch_smooth")
    def larch_smooth(
        ctx: Context,
        group_id: str,
        array_name: str = "mu",
        sigma: float = 1.0,
        form: str = "lorentzian",
        output_name: str | None = None,
    ) -> dict:
        """Smooth an array by convolving with a peak function.

        Args:
            group_id: ID of the group.
            array_name: Name of the array to smooth (e.g., "mu", "norm", "chi").
            sigma: Width parameter for the smoothing kernel.
            form: Kernel shape: "lorentzian", "gaussian", or "voigt".
            output_name: Name for the smoothed output array. Defaults to
                         "{array_name}_smooth".

        Returns:
            Info about the smoothed array.
        """
        session = _get_session(ctx)
        try:
            group = session.get_group(group_id)
        except KeyError as e:
            return {"error": str(e)}

        if not hasattr(group, array_name):
            return {"error": f"Array '{array_name}' not found in group '{group_id}'."}

        try:
            from larch.math import smooth

            y = getattr(group, array_name)
            # Need corresponding x array
            if array_name in ("mu", "norm", "flat", "dmude", "d2mude", "bkg"):
                x = group.energy if hasattr(group, "energy") else np.arange(len(y))
            elif array_name in ("chi",):
                x = group.k if hasattr(group, "k") else np.arange(len(y))
            else:
                x = np.arange(len(y))

            result = smooth(x, y, sigma=sigma, form=form)
            out_name = output_name or f"{array_name}_smooth"
            setattr(group, out_name, result)

            return {
                "group_id": group_id,
                "output_array": out_name,
                "shape": list(result.shape),
                "sigma": sigma,
                "form": form,
            }

        except Exception as e:
            return {"error": format_error("larch_smooth", e)}

    @mcp.tool(name="larch_deriv")
    def larch_deriv(
        ctx: Context,
        group_id: str,
        array_name: str = "mu",
        output_name: str | None = None,
    ) -> dict:
        """Compute the numerical derivative (gradient) of an array.

        Args:
            group_id: ID of the group.
            array_name: Name of the array to differentiate.
            output_name: Name for the output. Defaults to "d_{array_name}".

        Returns:
            Info about the derivative array.
        """
        session = _get_session(ctx)
        try:
            group = session.get_group(group_id)
        except KeyError as e:
            return {"error": str(e)}

        if not hasattr(group, array_name):
            return {"error": f"Array '{array_name}' not found in group '{group_id}'."}

        try:
            y = getattr(group, array_name)
            result = np.gradient(y)
            out_name = output_name or f"d_{array_name}"
            setattr(group, out_name, result)

            return {
                "group_id": group_id,
                "output_array": out_name,
                "shape": list(result.shape),
            }

        except Exception as e:
            return {"error": format_error("larch_deriv", e)}

    @mcp.tool(name="larch_interpolate")
    def larch_interpolate(
        ctx: Context,
        group_id: str,
        new_energy_min: float | None = None,
        new_energy_max: float | None = None,
        new_energy_step: float = 0.5,
        kind: str = "cubic",
    ) -> dict:
        """Interpolate spectrum onto a new uniform energy grid.

        Args:
            group_id: ID of the group with energy and mu arrays.
            new_energy_min: Start of new grid. Defaults to current min.
            new_energy_max: End of new grid. Defaults to current max.
            new_energy_step: Step size in eV. Default 0.5.
            kind: Interpolation method: "linear", "cubic", etc. Default "cubic".

        Returns:
            New grid info and number of points.
        """
        session = _get_session(ctx)
        try:
            group = session.get_group(group_id)
        except KeyError as e:
            return {"error": str(e)}

        if not hasattr(group, "energy"):
            return {"error": f"Group '{group_id}' has no 'energy' array."}

        try:
            from larch.math import interp

            emin = new_energy_min or float(group.energy.min())
            emax = new_energy_max or float(group.energy.max())
            new_energy = np.arange(emin, emax + new_energy_step / 2, new_energy_step)

            # Interpolate all 1D arrays that match energy length
            n_orig = len(group.energy)
            interpolated = []
            for attr in dir(group):
                if attr.startswith("_") or attr == "energy":
                    continue
                val = getattr(group, attr, None)
                if isinstance(val, np.ndarray) and val.shape == (n_orig,):
                    new_val = interp(group.energy, val, new_energy, kind=kind)
                    setattr(group, attr, new_val)
                    interpolated.append(attr)

            group.energy = new_energy

            return {
                "group_id": group_id,
                "n_points": len(new_energy),
                "energy_range": [float(new_energy.min()), float(new_energy.max())],
                "energy_step": new_energy_step,
                "interpolated_arrays": interpolated,
            }

        except Exception as e:
            return {"error": format_error("larch_interpolate", e)}

    @mcp.tool(name="larch_merge_groups")
    def larch_merge_groups(
        ctx: Context,
        group_ids: list[str],
        output_id: str = "merged",
        xarray: str = "energy",
        yarray: str = "mu",
    ) -> dict:
        """Merge/average multiple spectra into one.

        Interpolates all spectra onto a common energy grid and averages them.

        Args:
            group_ids: List of group IDs to merge.
            output_id: ID for the merged output group. Default "merged".
            xarray: X-axis array name. Default "energy".
            yarray: Y-axis array name to merge. Default "mu".

        Returns:
            Merged group info.
        """
        session = _get_session(ctx)
        groups = []
        for gid in group_ids:
            try:
                groups.append(session.get_group(gid))
            except KeyError as e:
                return {"error": str(e)}

        if len(groups) < 2:
            return {"error": "Need at least 2 groups to merge."}

        try:
            from larch.io import merge_groups

            merged = merge_groups(
                groups, xarray=xarray, yarray=yarray, kind="cubic"
            )
            gid = session.add_group(merged, group_id=output_id)

            return {
                "group_id": gid,
                "n_merged": len(groups),
                "n_points": (
                    len(merged.energy) if hasattr(merged, "energy") else None
                ),
                "energy_range": (
                    [float(merged.energy.min()), float(merged.energy.max())]
                    if hasattr(merged, "energy")
                    else None
                ),
                "has_yerr": hasattr(merged, f"{yarray}_std"),
            }

        except Exception as e:
            return {"error": format_error("larch_merge_groups", e)}

    @mcp.tool(name="larch_deglitch")
    def larch_deglitch(
        ctx: Context,
        group_id: str,
        energy_points: list[float],
        tolerance: float = 0.5,
    ) -> dict:
        """Remove glitch points from a spectrum by energy value.

        Removes data points near the specified energies.

        Args:
            group_id: ID of the group.
            energy_points: List of energy values (eV) where glitches occur.
            tolerance: Energy tolerance (eV) for matching glitch points.

        Returns:
            Number of points removed.
        """
        session = _get_session(ctx)
        try:
            group = session.get_group(group_id)
        except KeyError as e:
            return {"error": str(e)}

        if not hasattr(group, "energy"):
            return {"error": f"Group '{group_id}' has no 'energy' array."}

        try:
            energy = group.energy
            mask = np.ones(len(energy), dtype=bool)
            for ep in energy_points:
                mask &= np.abs(energy - ep) > tolerance

            n_removed = int(np.sum(~mask))
            n_orig = len(energy)

            # Remove points from all matching-length arrays
            for attr in dir(group):
                if attr.startswith("_"):
                    continue
                val = getattr(group, attr, None)
                if isinstance(val, np.ndarray) and val.shape == (n_orig,):
                    setattr(group, attr, val[mask])

            return {
                "group_id": group_id,
                "points_removed": n_removed,
                "points_remaining": int(np.sum(mask)),
            }

        except Exception as e:
            return {"error": format_error("larch_deglitch", e)}
