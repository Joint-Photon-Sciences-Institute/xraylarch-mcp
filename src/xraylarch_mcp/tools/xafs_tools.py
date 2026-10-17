"""XAFS analysis tools: normalization, background removal, Fourier transforms."""

from __future__ import annotations

from typing import Any

import numpy as np
from mcp.server.mcpserver import Context, MCPServer

from ..session import SessionManager
from ..util import format_error


def _get_session(ctx: Context) -> SessionManager:
    return ctx.request_context.lifespan_context["session"]


def register(mcp: MCPServer) -> None:
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
        clamp_lo: int = 0,
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
            clamp_lo: Clamp value at low-k (0=none, 1=moderate, 2=strong);
                      default 0, as in larch.
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
        line: str | None = None,
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
            line: Detected fluorescence line ("Ka", "La", "Lb", ...). Default:
                  "Ka" for a K edge, "La" for L3, "Lb" for L2 and L1.
            anginp: Incident beam angle in degrees. Default 45.
            angout: Exit beam angle in degrees. Default 45.

        Returns:
            The new group holding the corrected spectrum (its mu is the
            corrected mu), the normalization used, alpha, and the normalized
            peak height before and after. FLUO scales mu by
            alpha / (alpha + 1 - norm), so it fails where the normalized
            spectrum reaches 1 + alpha; the tool then returns an error, and
            a warning when the spectrum comes close enough to more than
            triple the peak.
        """
        session = _get_session(ctx)
        try:
            group = session.get_group(group_id)
        except KeyError as e:
            return {"error": str(e)}

        if not hasattr(group, "energy") or not hasattr(group, "mu"):
            return {"error": f"Group '{group_id}' needs energy and mu arrays."}

        if line is None:
            line = {"K": "Ka", "L3": "La", "L2": "Lb", "L1": "Lb"}.get(edge.upper())
            if line is None:
                return {"error": f"No default fluorescence line for the {edge} edge: set line."}

        try:
            from larch import Group
            from larch.xafs import fluo_corr, pre_edge
            from larch.xray import material_mu, xray_edge, xray_line

            # fluo_corr looks for the group's normalization in
            # pre_edge_details.call_args, which pre_edge never sets, so it
            # would use its own ranges. Pass the group's ranges explicitly.
            normalized_here = not hasattr(group, "pre_edge_details")
            if normalized_here:
                pre_edge(group)
            details = group.pre_edge_details
            normalization = {
                "e0": float(group.e0),
                "pre1": float(details.pre1),
                "pre2": float(details.pre2),
                "norm1": float(details.norm1),
                "norm2": float(details.norm2),
                "nnorm": int(details.nnorm),
            }

            fluo_corr(
                group.energy,
                group.mu,
                formula,
                elem,
                group=group,
                edge=edge,
                line=line,
                anginp=anginp,
                angout=angout,
                **normalization,
            )

            # alpha as fluo_corr computes it, for the result and the check below
            ang_corr = np.sin(max(1.0e-7, np.deg2rad(anginp))) / np.sin(
                max(1.0e-7, np.deg2rad(angout))
            )
            e_edge = xray_edge(elem, edge).energy
            mu_fluor, mu_below, mu_above = material_mu(
                formula,
                np.array([xray_line(elem, line).energy, e_edge - 10.0, e_edge + 10.0]),
                density=1,
            )
            alpha = float((mu_fluor * ang_corr + mu_below) / (mu_above - mu_below))

            # Where the normalized spectrum reaches 1 + alpha the correction
            # factor alpha / (alpha + 1 - norm) passes through infinity and
            # changes sign.
            if not np.all(np.isfinite(group.mu_corr)) or np.any(group.mu_corr * group.mu < 0):
                return {
                    "error": (
                        f"The FLUO correction diverges for '{group_id}': its normalized "
                        f"spectrum reaches {float(group.norm.max()):.3f}, above "
                        f"1 + alpha = {1 + alpha:.3f} for {formula}. In the FLUO model a "
                        "self-absorbed fluorescence spectrum always stays below 1 + alpha, "
                        "so check the formula, elem, edge and line, and that this is "
                        "fluorescence data. Glitches above the edge can also cause it."
                    )
                }

            # larch writes mu_corr / norm_corr; the other tools read mu, so the
            # corrected spectrum goes into a group of its own.
            corrected = Group(
                energy=group.energy.copy(), mu=group.mu_corr.copy(), e0=float(group.e0)
            )
            new_id = session.add_group(corrected, group_id=f"{group_id}_fluo_corr")

            result = {
                "group_id": new_id,
                "source_group_id": group_id,
                "formula": formula,
                "elem": elem,
                "edge": edge,
                "line": line,
                "alpha": alpha,
                "normalization": normalization,
                "norm_max_before": float(group.norm.max()),
                "norm_max_corrected": float(group.norm_corr.max()),
            }
            if normalized_here:
                result["note"] = (
                    f"Group '{group_id}' was not normalized, so it was normalized "
                    "with the default pre_edge settings first."
                )
            # Just below 1 + alpha the correction does not change sign but is
            # still dominated by the singularity. Strong self-absorption raises
            # a white line by up to about 2.5 times.
            if result["norm_max_corrected"] > 3 * result["norm_max_before"]:
                result["warning"] = (
                    f"The correction raises the normalized peak from "
                    f"{result['norm_max_before']:.3f} to {result['norm_max_corrected']:.3f}: "
                    f"the spectrum comes within {1 + alpha - result['norm_max_before']:.3f} "
                    f"of 1 + alpha = {1 + alpha:.3f}, where FLUO diverges, so the corrected "
                    "spectrum is unreliable. Check the formula, elem, edge and line, and "
                    "that this is fluorescence data."
                )
            return result

        except Exception as e:
            return {"error": format_error("larch_fluo_corr", e)}

    @mcp.tool(name="larch_mback_norm")
    def larch_mback_norm(
        ctx: Context,
        group_id: str,
        z: int | None = None,
        edge: str = "K",
        e0: float | None = None,
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

        Returns:
            Normalization results. norm and edge_step now hold the MBACK
            normalization; the pre_edge ones are kept in norm_poly and
            edge_step_poly.
        """
        session = _get_session(ctx)
        try:
            group = session.get_group(group_id)
        except KeyError as e:
            return {"error": str(e)}

        try:
            from larch.xafs import mback_norm

            kwargs: dict[str, Any] = {"edge": edge}
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
                "edge_step_poly": (
                    float(group.edge_step_poly) if hasattr(group, "edge_step_poly") else None
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
            The new group holding the rebinned spectrum, and its grid.
        """
        session = _get_session(ctx)
        try:
            group = session.get_group(group_id)
        except KeyError as e:
            return {"error": str(e)}

        if not hasattr(group, "energy") or not hasattr(group, "mu"):
            return {"error": f"Group '{group_id}' needs energy and mu arrays."}

        try:
            from larch import Group
            from larch.xafs import find_e0, rebin_xafs

            # rebin_xafs falls back to group.e0 but not to find_e0
            if e0 is None:
                e0 = getattr(group, "e0", None)
            if e0 is None:
                e0 = find_e0(group.energy, group.mu)

            kwargs: dict[str, Any] = {"e0": float(e0), "exafs_kstep": exafs_kstep}
            if pre1 is not None:
                kwargs["pre1"] = pre1
            if pre2 is not None:
                kwargs["pre2"] = pre2
            if pre_step is not None:
                kwargs["pre_step"] = pre_step
            if xanes_step is not None:
                kwargs["xanes_step"] = xanes_step

            rebin_xafs(group, **kwargs)

            # larch leaves the group's arrays alone and writes the result to
            # group.rebinned, which the other tools cannot reach. delta_mu is
            # the spread of mu within each bin (NaN for single-point bins).
            rebinned = Group(
                energy=np.asarray(group.rebinned.energy, dtype=float),
                mu=np.asarray(group.rebinned.mu, dtype=float),
                delta_mu=np.asarray(group.rebinned.delta_mu, dtype=float),
                e0=float(e0),
            )
            new_id = session.add_group(rebinned, group_id=f"{group_id}_rebinned")

            return {
                "group_id": new_id,
                "source_group_id": group_id,
                "e0": float(e0),
                "n_points_before": len(group.energy),
                "n_points": len(rebinned.energy),
                "energy_range": [
                    float(rebinned.energy.min()),
                    float(rebinned.energy.max()),
                ],
            }

        except Exception as e:
            return {"error": format_error("larch_rebin", e)}
