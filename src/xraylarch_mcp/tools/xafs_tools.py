"""XAFS analysis tools: normalization, background removal, Fourier transforms."""

from __future__ import annotations

from typing import Any

import numpy as np
from mcp.server.fastmcp import Context, FastMCP

from ..session import SessionManager
from ..util import format_error


def _get_session(ctx: Context) -> SessionManager:
    return ctx.request_context.lifespan_context["session"]


def register(mcp: FastMCP) -> None:
    @mcp.tool(name="larch_normalize")
    def larch_normalize(
        ctx: Context,
        group_id: str,
        e0: float | None = None,
        pre1: float | None = None,
        pre2: float | None = None,
        norm1: float | None = None,
        norm2: float | None = None,
        nnorm: int | None = None,
        nvict: int = 0,
    ) -> dict:
        """Normalize an X-ray absorption spectrum (pre-edge subtraction + normalization).

        Wraps larch's pre_edge() function. Determines E0, fits pre-edge line and
        post-edge polynomial, computes normalized mu(E), flattened mu(E), and
        the derivative dmu/dE.

        Args:
            group_id: ID of the loaded spectrum group.
            e0: Edge energy in eV. If None, auto-detected from max of derivative.
            pre1: Lower bound of pre-edge fit range relative to E0 (eV). E.g., -150.
            pre2: Upper bound of pre-edge fit range relative to E0 (eV). E.g., -30.
            norm1: Lower bound of post-edge fit range relative to E0 (eV). E.g., 50.
            norm2: Upper bound of post-edge fit range relative to E0 (eV). E.g., 300.
            nnorm: Degree of post-edge normalization polynomial (0, 1, 2, or 3).
            nvict: Energy exponent for pre-edge fit (0=no correction).

        Returns:
            E0, edge_step, normalization details.
        """
        session = _get_session(ctx)
        try:
            group = session.get_group(group_id)
        except KeyError as e:
            return {"error": str(e)}

        if not hasattr(group, "energy") or not hasattr(group, "mu"):
            return {
                "error": f"Group '{group_id}' missing 'energy' and/or 'mu' arrays. "
                f"Available arrays: {[a for a in dir(group) if not a.startswith('_') and isinstance(getattr(group, a, None), np.ndarray)]}"
            }

        try:
            from larch.xafs import pre_edge

            kwargs: dict[str, Any] = {}
            if e0 is not None:
                kwargs["e0"] = e0
            if pre1 is not None:
                kwargs["pre1"] = pre1
            if pre2 is not None:
                kwargs["pre2"] = pre2
            if norm1 is not None:
                kwargs["norm1"] = norm1
            if norm2 is not None:
                kwargs["norm2"] = norm2
            if nnorm is not None:
                kwargs["nnorm"] = nnorm
            kwargs["nvict"] = nvict

            pre_edge(group, **kwargs)

            result: dict[str, Any] = {
                "group_id": group_id,
                "e0": float(group.e0),
                "edge_step": float(group.edge_step),
                "atsym": getattr(group, "atsym", None),
                "edge": getattr(group, "edge", None),
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

            result["norm_range"] = [float(group.norm.min()), float(group.norm.max())]
            return result

        except Exception as e:
            hints = None
            if "edge" in str(e).lower() or "e0" in str(e).lower():
                hints = (
                    "E0 auto-detection may have failed. The data may not contain "
                    "an absorption edge in the provided energy range. "
                    "Try setting e0 manually."
                )
            return {"error": format_error("larch_normalize", e, hints)}

    @mcp.tool(name="larch_autobk")
    def larch_autobk(
        ctx: Context,
        group_id: str,
        rbkg: float = 1.0,
        kweight: int = 2,
        kmin: float = 0.0,
        kmax: float | None = None,
        nknots: int | None = None,
        clamp_lo: int = 1,
        clamp_hi: int = 1,
    ) -> dict:
        """Extract EXAFS chi(k) using the AUTOBK algorithm.

        Removes the smooth atomic background from mu(E) to isolate
        the EXAFS oscillations chi(k).

        Args:
            group_id: ID of the loaded spectrum group (must be normalized first).
            rbkg: Distance (in Angstroms) for the background spline. Should be
                  less than the nearest-neighbor distance. Default 1.0.
            kweight: k-weighting for background fit. Default 2.
            kmin: Minimum k value for the fit.
            kmax: Maximum k value. If None, auto-determined from data.
            nknots: Number of spline knots. If None, auto-determined.
            clamp_lo: Clamp value at low-k (0=none, 1=moderate, 2=strong).
            clamp_hi: Clamp value at high-k.

        Returns:
            k-range achieved, rbkg used, chi statistics.
        """
        session = _get_session(ctx)
        try:
            group = session.get_group(group_id)
        except KeyError as e:
            return {"error": str(e)}

        if not hasattr(group, "energy") or not hasattr(group, "mu"):
            return {
                "error": f"Group '{group_id}' needs energy and mu arrays. Run larch_normalize first."
            }

        try:
            from larch.xafs import autobk

            kwargs: dict[str, Any] = {
                "rbkg": rbkg,
                "kweight": kweight,
                "kmin": kmin,
                "clamp_lo": clamp_lo,
                "clamp_hi": clamp_hi,
            }
            if kmax is not None:
                kwargs["kmax"] = kmax
            if nknots is not None:
                kwargs["nknots"] = nknots

            autobk(group, **kwargs)

            return {
                "group_id": group_id,
                "k_range": [float(group.k.min()), float(group.k.max())],
                "k_points": len(group.k),
                "rbkg": float(group.rbkg),
                "chi_max": float(np.max(np.abs(group.chi))),
                "chi_mean": float(np.mean(np.abs(group.chi))),
            }

        except Exception as e:
            hints = None
            if "rbkg" in str(e).lower():
                hints = (
                    "rbkg may be too large for this system. "
                    "Try a smaller value (e.g., 0.8 or 1.0)."
                )
            return {"error": format_error("larch_autobk", e, hints)}

    @mcp.tool(name="larch_xftf")
    def larch_xftf(
        ctx: Context,
        group_id: str,
        kmin: float = 2.0,
        kmax: float | None = None,
        dk: float = 2.0,
        kweight: int = 2,
        window: str = "kaiser",
    ) -> dict:
        """Forward Fourier transform: chi(k) -> chi(R).

        Transforms EXAFS data from k-space to R-space to reveal
        the radial distribution of neighboring atoms.

        Args:
            group_id: ID of the group (must have chi(k) from larch_autobk).
            kmin: Minimum k for the FT window (Angstrom^-1).
            kmax: Maximum k for the FT window. If None, uses max available.
            dk: Window sill width (Angstrom^-1).
            kweight: k-weighting power (typically 1, 2, or 3).
            window: FT window type: "kaiser", "hanning", "parzen", "welch", "gaussian", "sine".

        Returns:
            R-range, first-shell peak position estimate.
        """
        session = _get_session(ctx)
        try:
            group = session.get_group(group_id)
        except KeyError as e:
            return {"error": str(e)}

        if not hasattr(group, "k") or not hasattr(group, "chi"):
            return {
                "error": f"Group '{group_id}' has no chi(k). Run larch_autobk first."
            }

        try:
            from larch.xafs import xftf

            kwargs: dict[str, Any] = {
                "kmin": kmin,
                "dk": dk,
                "kweight": kweight,
                "window": window,
            }
            if kmax is not None:
                kwargs["kmax"] = kmax
            else:
                kwargs["kmax"] = float(group.k.max())

            xftf(group, **kwargs)

            # Find first-shell peak
            r = group.r
            chir_mag = group.chir_mag
            # Look for peak in reasonable range (1.0 - 3.5 A)
            mask = (r >= 1.0) & (r <= 3.5)
            if mask.any():
                peak_idx = np.argmax(chir_mag[mask])
                first_shell_r = float(r[mask][peak_idx])
            else:
                first_shell_r = None

            return {
                "group_id": group_id,
                "r_range": [float(r.min()), float(r.max())],
                "r_points": len(r),
                "kmin_used": kmin,
                "kmax_used": kwargs["kmax"],
                "kweight": kweight,
                "window": window,
                "first_shell_peak_r": first_shell_r,
                "chir_mag_max": float(chir_mag.max()),
            }

        except Exception as e:
            return {"error": format_error("larch_xftf", e)}

    @mcp.tool(name="larch_xftr")
    def larch_xftr(
        ctx: Context,
        group_id: str,
        rmin: float = 0.0,
        rmax: float = 5.0,
        dr: float = 0.1,
        window: str = "hanning",
    ) -> dict:
        """Back Fourier transform: chi(R) -> chi(q).

        Filters specific R-range shells and transforms back to k-space.

        Args:
            group_id: ID of the group (must have chi(R) from larch_xftf).
            rmin: Minimum R for the back-transform window (Angstrom).
            rmax: Maximum R for the back-transform window (Angstrom).
            dr: Window sill width (Angstrom).
            window: Window type: "hanning", "kaiser", "parzen", "welch", etc.

        Returns:
            q-range, filtered chi statistics.
        """
        session = _get_session(ctx)
        try:
            group = session.get_group(group_id)
        except KeyError as e:
            return {"error": str(e)}

        if not hasattr(group, "r") or not hasattr(group, "chir"):
            return {
                "error": f"Group '{group_id}' has no chi(R). Run larch_xftf first."
            }

        try:
            from larch.xafs import xftr

            xftr(group, rmin=rmin, rmax=rmax, dr=dr, window=window)

            return {
                "group_id": group_id,
                "q_range": [float(group.q.min()), float(group.q.max())],
                "q_points": len(group.q),
                "rmin": rmin,
                "rmax": rmax,
                "window": window,
                "chiq_mag_max": float(group.chiq_mag.max()),
            }

        except Exception as e:
            return {"error": format_error("larch_xftr", e)}

    @mcp.tool(name="larch_fluo_corr")
    def larch_fluo_corr(
        ctx: Context,
        group_id: str,
        formula: str,
        elem: str,
        edge: str = "K",
        anginp: float = 45.0,
        angout: float = 45.0,
    ) -> dict:
        """Apply fluorescence self-absorption correction (FLUO algorithm).

        Corrects for over-absorption in fluorescence-detected XAFS data.

        Args:
            group_id: ID of the group with energy and mu arrays.
            formula: Chemical formula of the sample (e.g., "FeO", "Fe2O3").
            elem: Absorbing element symbol (e.g., "Fe").
            edge: Absorption edge ("K", "L3", etc.). Default "K".
            anginp: Incident beam angle in degrees. Default 45.
            angout: Exit beam angle in degrees. Default 45.

        Returns:
            Confirmation and summary of corrected data.
        """
        session = _get_session(ctx)
        try:
            group = session.get_group(group_id)
        except KeyError as e:
            return {"error": str(e)}

        try:
            from larch.xafs import fluo_corr

            fluo_corr(
                group,
                formula=formula,
                elem=elem,
                edge=edge,
                anginp=anginp,
                angout=angout,
            )

            return {
                "group_id": group_id,
                "correction_applied": True,
                "formula": formula,
                "elem": elem,
                "edge": edge,
            }

        except Exception as e:
            return {"error": format_error("larch_fluo_corr", e)}

    @mcp.tool(name="larch_mback_norm")
    def larch_mback_norm(
        ctx: Context,
        group_id: str,
        z: int | None = None,
        edge: str = "K",
        e0: float | None = None,
        order: int = 3,
    ) -> dict:
        """MBACK normalization (alternative to standard pre_edge normalization).

        Uses the MBACK algorithm which matches the post-edge to tabulated
        absorption coefficients. Useful for problematic data where standard
        normalization fails.

        Args:
            group_id: ID of the group with energy and mu.
            z: Atomic number of the absorbing element (if not auto-detected).
            edge: Absorption edge. Default "K".
            e0: Edge energy in eV. If None, auto-detected.
            order: Polynomial order for the fit. Default 3.

        Returns:
            Normalization results.
        """
        session = _get_session(ctx)
        try:
            group = session.get_group(group_id)
        except KeyError as e:
            return {"error": str(e)}

        try:
            from larch.xafs import mback_norm

            kwargs: dict[str, Any] = {"edge": edge, "order": order}
            if z is not None:
                kwargs["z"] = z
            if e0 is not None:
                kwargs["e0"] = e0

            mback_norm(group, **kwargs)

            return {
                "group_id": group_id,
                "e0": float(group.e0) if hasattr(group, "e0") else None,
                "edge_step": (
                    float(group.edge_step) if hasattr(group, "edge_step") else None
                ),
                "norm_range": (
                    [float(group.norm.min()), float(group.norm.max())]
                    if hasattr(group, "norm")
                    else None
                ),
            }

        except Exception as e:
            return {"error": format_error("larch_mback_norm", e)}

    @mcp.tool(name="larch_estimate_noise")
    def larch_estimate_noise(ctx: Context, group_id: str) -> dict:
        """Estimate noise level in chi(k) and chi(R).

        Must have run autobk and xftf first.

        Args:
            group_id: ID of the group with chi(k).

        Returns:
            epsilon_k (noise in k-space) and epsilon_r (noise in R-space).
        """
        session = _get_session(ctx)
        try:
            group = session.get_group(group_id)
        except KeyError as e:
            return {"error": str(e)}

        if not hasattr(group, "chi"):
            return {
                "error": f"Group '{group_id}' has no chi(k). Run larch_autobk first."
            }

        try:
            from larch.xafs import estimate_noise

            estimate_noise(group)

            return {
                "group_id": group_id,
                "epsilon_k": float(group.epsilon_k),
                "epsilon_r": float(group.epsilon_r),
            }

        except Exception as e:
            return {"error": format_error("larch_estimate_noise", e)}

    @mcp.tool(name="larch_cauchy_wavelet")
    def larch_cauchy_wavelet(
        ctx: Context,
        group_id: str,
        kweight: int = 2,
        rmax_out: float = 10.0,
        nfft: int = 2048,
    ) -> dict:
        """Cauchy Wavelet Transform of EXAFS chi(k).

        Computes the continuous Cauchy wavelet transform, providing
        simultaneous k- and R-space resolution. Useful for identifying
        which k-ranges contribute to specific R-space features.

        Must have run larch_autobk first to extract chi(k).

        Args:
            group_id: ID of the group with chi(k) from larch_autobk.
            kweight: k-weighting power (0, 1, 2, or 3). Default 2.
            rmax_out: Maximum R for output (Angstroms). Default 10.
            nfft: FFT size. Default 2048.

        Returns:
            k-range, R-range, shape of wavelet magnitude array.
        """
        session = _get_session(ctx)
        try:
            group = session.get_group(group_id)
        except KeyError as e:
            return {"error": str(e)}

        if not hasattr(group, "k") or not hasattr(group, "chi"):
            return {
                "error": f"Group '{group_id}' has no chi(k). Run larch_autobk first."
            }

        try:
            from larch.xafs.cauchy_wavelet import cauchy_wavelet

            cauchy_wavelet(
                group.k, group.chi,
                group=group,
                kweight=kweight,
                rmax_out=rmax_out,
                nfft=nfft,
            )

            return {
                "group_id": group_id,
                "k_range": [float(group.k.min()), float(group.k.max())],
                "r_range": [float(group.wcauchy_r.min()), float(group.wcauchy_r.max())],
                "shape": list(group.wcauchy_mag.shape),
                "kweight": kweight,
                "rmax_out": rmax_out,
                "wcauchy_mag_max": float(group.wcauchy_mag.max()),
            }

        except Exception as e:
            return {"error": format_error("larch_cauchy_wavelet", e)}

    @mcp.tool(name="larch_rebin")
    def larch_rebin(
        ctx: Context,
        group_id: str,
        e0: float | None = None,
        pre1: float | None = None,
        pre2: float | None = None,
        pre_step: float | None = None,
        xanes_step: float | None = None,
        exafs_kstep: float = 0.05,
    ) -> dict:
        """Rebin XAFS data to a standard 3-region energy grid.

        Creates a uniform grid with coarse steps in the pre-edge,
        fine steps through the XANES, and uniform k-steps in the EXAFS.

        Args:
            group_id: ID of the group with energy and mu.
            e0: Edge energy. If None, auto-detected.
            pre1: Start of pre-edge region relative to E0.
            pre2: End of pre-edge / start of XANES relative to E0.
            pre_step: Energy step in pre-edge region.
            xanes_step: Energy step in XANES region.
            exafs_kstep: k-step in EXAFS region. Default 0.05.

        Returns:
            New grid info.
        """
        session = _get_session(ctx)
        try:
            group = session.get_group(group_id)
        except KeyError as e:
            return {"error": str(e)}

        try:
            from larch.xafs import rebin_xafs

            kwargs: dict[str, Any] = {"exafs_kstep": exafs_kstep}
            if e0 is not None:
                kwargs["e0"] = e0
            if pre1 is not None:
                kwargs["pre1"] = pre1
            if pre2 is not None:
                kwargs["pre2"] = pre2
            if pre_step is not None:
                kwargs["pre_step"] = pre_step
            if xanes_step is not None:
                kwargs["xanes_step"] = xanes_step

            rebin_xafs(group, **kwargs)

            return {
                "group_id": group_id,
                "n_points": len(group.energy),
                "energy_range": [
                    float(group.energy.min()),
                    float(group.energy.max()),
                ],
            }

        except Exception as e:
            return {"error": format_error("larch_rebin", e)}
