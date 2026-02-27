"""Fitting tools: linear combination fitting, PCA, FEFFIT."""

from __future__ import annotations

from typing import Any

import numpy as np
from mcp.server.fastmcp import Context, FastMCP

from ..session import SessionManager
from ..util import format_error


def _get_session(ctx: Context) -> SessionManager:
    return ctx.request_context.lifespan_context["session"]


def register(mcp: FastMCP) -> None:
    @mcp.tool(name="larch_lcf")
    def larch_lcf(
        ctx: Context,
        group_id: str,
        standard_group_ids: list[str],
        xmin: float = -float("inf"),
        xmax: float = float("inf"),
        arrayname: str = "norm",
        sum_to_one: bool = True,
        vary_e0: bool = False,
    ) -> dict:
        """Linear combination fitting of a spectrum against standards.

        Fits the unknown spectrum as a weighted sum of known standard spectra.
        Commonly used for XANES fingerprinting / phase identification.

        Args:
            group_id: ID of the unknown spectrum group.
            standard_group_ids: List of IDs of standard/reference spectrum groups.
            xmin: Minimum energy (or k) for the fit range.
            xmax: Maximum energy (or k) for the fit range.
            arrayname: Which array to fit: "norm", "flat", "dmude", "chi", etc.
            sum_to_one: Constrain weights to sum to 1. Default True.
            vary_e0: Allow E0 shifts for each standard. Default False.

        Returns:
            Weights for each standard, R-factor, fit statistics.
        """
        session = _get_session(ctx)
        try:
            group = session.get_group(group_id)
            standards = [session.get_group(sid) for sid in standard_group_ids]
        except KeyError as e:
            return {"error": str(e)}

        try:
            from larch.math import lincombo_fit

            result = lincombo_fit(
                group,
                standards,
                arrayname=arrayname,
                xmin=xmin,
                xmax=xmax,
                sum_to_one=sum_to_one,
                vary_e0=vary_e0,
            )

            # Extract results
            weights = {}
            for i, sid in enumerate(standard_group_ids):
                w_attr = f"weight_{i}"
                if hasattr(result, w_attr):
                    w = getattr(result, w_attr)
                    weights[sid] = float(w) if isinstance(w, (int, float)) else None

            # Try to get weights from result params
            if not any(v is not None for v in weights.values()):
                if hasattr(result, "weights"):
                    for i, sid in enumerate(standard_group_ids):
                        if i < len(result.weights):
                            weights[sid] = float(result.weights[i])

            return {
                "group_id": group_id,
                "weights": weights,
                "rfactor": float(result.rfactor) if hasattr(result, "rfactor") else None,
                "chisqr": float(result.chisqr) if hasattr(result, "chisqr") else None,
                "redchi": float(result.redchi) if hasattr(result, "redchi") else None,
                "arrayname": arrayname,
                "xmin": xmin,
                "xmax": xmax,
                "sum_to_one": sum_to_one,
            }

        except Exception as e:
            return {"error": format_error("larch_lcf", e)}

    @mcp.tool(name="larch_pca")
    def larch_pca(
        ctx: Context,
        group_ids: list[str],
        arrayname: str = "norm",
        xmin: float = -float("inf"),
        xmax: float = float("inf"),
    ) -> dict:
        """Principal Component Analysis on a set of spectra.

        Decomposes a collection of spectra into orthogonal components
        to identify the number of distinct species.

        Args:
            group_ids: List of spectrum group IDs to include.
            arrayname: Array to analyze ("norm", "flat", "chi", etc.).
            xmin: Minimum x-value for the analysis range.
            xmax: Maximum x-value for the analysis range.

        Returns:
            Eigenvalues, variance explained, suggested number of components.
        """
        session = _get_session(ctx)
        try:
            groups = [session.get_group(gid) for gid in group_ids]
        except KeyError as e:
            return {"error": str(e)}

        if len(groups) < 3:
            return {"error": "PCA requires at least 3 spectra."}

        try:
            from larch.math import pca_train

            pca_model = pca_train(
                groups, arrayname=arrayname, xmin=xmin, xmax=xmax
            )

            eigenvalues = pca_model.eigenvalues
            variance = pca_model.variances

            # Suggest number of components using eigenvalue > 1 criterion
            # or where cumulative variance > 95%
            cumvar = np.cumsum(variance)
            n_suggested = int(np.searchsorted(cumvar, 0.95)) + 1
            n_suggested = min(n_suggested, len(eigenvalues))

            # Store PCA model on session for later use
            session._pca_model = pca_model

            return {
                "n_spectra": len(groups),
                "n_components": len(eigenvalues),
                "eigenvalues": [float(e) for e in eigenvalues[:10]],
                "variance_explained": [float(v) for v in variance[:10]],
                "cumulative_variance": [float(c) for c in cumvar[:10]],
                "suggested_n_components": n_suggested,
            }

        except Exception as e:
            return {"error": format_error("larch_pca", e)}

    @mcp.tool(name="larch_feffit")
    def larch_feffit(
        ctx: Context,
        group_id: str,
        paths: list[dict],
        kmin: float = 2.0,
        kmax: float = 15.0,
        kweight: int = 2,
        dk: float = 4.0,
        window: str = "kaiser",
        rmin: float = 1.0,
        rmax: float = 4.0,
        fitspace: str = "r",
    ) -> dict:
        """FEFFIT: fit EXAFS data with theoretical scattering paths from FEFF.

        Each path needs a feffNNNN.dat file and parameters for S02, sigma2,
        deltar, and E0 shift.

        Args:
            group_id: ID of the data group with chi(k).
            paths: List of path configurations. Each is a dict with:
                - "filename": path to feffNNNN.dat file
                - "label": optional label for this path
                - "s02": S0^2 amplitude factor (float or "guess:value")
                - "sigma2": Debye-Waller factor (float or "guess:value")
                - "deltar": path length change (float or "guess:value")
                - "e0": energy shift (float or "guess:value")
                - "degen": degeneracy override (optional)
            kmin: Minimum k for transform.
            kmax: Maximum k for transform.
            kweight: k-weighting.
            dk: Window sill width.
            window: FT window type.
            rmin: Minimum R for fit range.
            rmax: Maximum R for fit range.
            fitspace: Fitting space: "r" (default), "k", or "q".

        Returns:
            Fit statistics, best-fit parameters, R-factor.
        """
        session = _get_session(ctx)
        try:
            group = session.get_group(group_id)
        except KeyError as e:
            return {"error": str(e)}

        if not hasattr(group, "k") or not hasattr(group, "chi"):
            return {"error": f"Group '{group_id}' needs chi(k). Run larch_autobk first."}

        try:
            from larch.fitting import param, param_group
            from larch.xafs import (
                FeffitDataSet,
                FeffPathGroup,
                TransformGroup,
                feffit,
                feffit_report,
            )

            # Build transform
            trans = TransformGroup(
                kmin=kmin,
                kmax=kmax,
                kweight=kweight,
                dk=dk,
                window=window,
                rmin=rmin,
                rmax=rmax,
                fitspace=fitspace,
            )

            # Build parameter group and paths
            params_dict = {}
            feff_paths = []

            for i, pconf in enumerate(paths):
                filename = pconf.get("filename", "")
                label = pconf.get("label", f"path_{i}")

                path_kwargs: dict[str, Any] = {}
                for pname in ("s02", "sigma2", "deltar", "e0"):
                    val = pconf.get(pname)
                    if val is None:
                        continue
                    if isinstance(val, str) and val.startswith("guess:"):
                        guess_val = float(val.split(":")[1])
                        p = param(guess_val, vary=True)
                        param_name = f"{pname}_{i}"
                        params_dict[param_name] = p
                        path_kwargs[pname] = p
                    else:
                        path_kwargs[pname] = float(val)

                if pconf.get("degen") is not None:
                    path_kwargs["degen"] = float(pconf["degen"])

                fp = FeffPathGroup(filename=filename, label=label, **path_kwargs)
                feff_paths.append(fp)

            pgroup = param_group(**params_dict)

            dataset = FeffitDataSet(
                data=group, paths=feff_paths, transform=trans
            )

            result = feffit(pgroup, [dataset])

            # Extract results
            report = feffit_report(result)

            # Build parameter results
            param_results = {}
            for name, p in params_dict.items():
                param_results[name] = {
                    "value": float(p.value),
                    "stderr": float(p.stderr) if p.stderr is not None else None,
                    "vary": p.vary,
                }

            return {
                "group_id": group_id,
                "rfactor": float(result.rfactor) if hasattr(result, "rfactor") else None,
                "chi_square": float(result.chi_square) if hasattr(result, "chi_square") else None,
                "chi_reduced": float(result.chi_reduced) if hasattr(result, "chi_reduced") else None,
                "n_independent": float(result.n_independent) if hasattr(result, "n_independent") else None,
                "n_variables": int(result.nvarys) if hasattr(result, "nvarys") else None,
                "parameters": param_results,
                "fit_report": report,
            }

        except Exception as e:
            return {"error": format_error("larch_feffit", e)}

    @mcp.tool(name="larch_peak_fit")
    def larch_peak_fit(
        ctx: Context,
        group_id: str,
        xarray: str = "energy",
        yarray: str = "norm",
        model: str = "voigt",
        background: str | None = "linear",
        xmin: float | None = None,
        xmax: float | None = None,
    ) -> dict:
        """Fit a peak in spectral data (e.g., pre-edge peaks, emission lines).

        Args:
            group_id: ID of the group.
            xarray: X-axis array name. Default "energy".
            yarray: Y-axis array name. Default "norm".
            model: Peak model: "gaussian", "lorentzian", "voigt", "pvoigt",
                   "pearson7", "students_t". Default "voigt".
            background: Background model: "linear", "constant", None. Default "linear".
            xmin: Minimum x for fit range.
            xmax: Maximum x for fit range.

        Returns:
            Peak parameters (center, amplitude, sigma, fwhm), fit statistics.
        """
        session = _get_session(ctx)
        try:
            group = session.get_group(group_id)
        except KeyError as e:
            return {"error": str(e)}

        if not hasattr(group, xarray) or not hasattr(group, yarray):
            return {
                "error": f"Group '{group_id}' missing '{xarray}' or '{yarray}'."
            }

        try:
            from larch.math import fit_peak

            x = np.array(getattr(group, xarray), dtype=float)
            y = np.array(getattr(group, yarray), dtype=float)

            # Apply range
            if xmin is not None or xmax is not None:
                mask = np.ones(len(x), dtype=bool)
                if xmin is not None:
                    mask &= x >= xmin
                if xmax is not None:
                    mask &= x <= xmax
                x = x[mask]
                y = y[mask]

            result = fit_peak(x, y, model, background=background)

            # Extract parameters
            params = {}
            if hasattr(result, "params"):
                for pname, pval in result.params.items():
                    params[pname] = {
                        "value": float(pval.value),
                        "stderr": (
                            float(pval.stderr) if pval.stderr is not None else None
                        ),
                    }

            return {
                "group_id": group_id,
                "model": model,
                "background": background,
                "parameters": params,
                "chisqr": float(result.chisqr) if hasattr(result, "chisqr") else None,
                "redchi": float(result.redchi) if hasattr(result, "redchi") else None,
            }

        except Exception as e:
            return {"error": format_error("larch_peak_fit", e)}
