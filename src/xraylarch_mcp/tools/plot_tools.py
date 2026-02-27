"""Plot tools: generate publication-quality plots of spectral data."""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mcp.server.fastmcp import Context, FastMCP

from ..session import SessionManager


def _get_session(ctx: Context) -> SessionManager:
    return ctx.request_context.lifespan_context["session"]


# Axis labels for different plot types
_AXIS_LABELS = {
    "mu": ("Energy (eV)", r"$\mu$(E)"),
    "norm": ("Energy (eV)", r"Normalized $\mu$(E)"),
    "flat": ("Energy (eV)", r"Flattened $\mu$(E)"),
    "dmude": ("Energy (eV)", r"d$\mu$/dE (eV$^{-1}$)"),
    "chi_k": (r"k ($\AA^{-1}$)", None),  # y-label depends on kweight
    "chi_r": (r"R ($\AA$)", None),
    "chi_r_mag": (r"R ($\AA$)", None),
    "chi_q": (r"q ($\AA^{-1}$)", None),
    "cauchy_wavelet": (r"k ($\AA^{-1}$)", r"R ($\AA$)"),
}


def _chi_ylabel(kweight: int) -> str:
    if kweight == 0:
        return r"$\chi$(k) ($\AA^{-1}$)"
    return rf"$\chi$(k)$\cdot$k$^{kweight}$ ($\AA^{{-{kweight + 1}}}$)"


def _chir_ylabel(kweight: int) -> str:
    return rf"|$\chi$(R)| ($\AA^{{-{kweight + 2}}}$)"


def _chiq_ylabel(kweight: int) -> str:
    if kweight == 0:
        return r"|$\chi$(q)| ($\AA^{-1}$)"
    return rf"|$\chi$(q)$\cdot$q$^{kweight}$| ($\AA^{{-{kweight + 1}}}$)"


def _get_plot_data(group: Any, plot_type: str, kweight: int = 2) -> tuple:
    """Extract x, y data and labels for a given plot type."""
    if plot_type == "mu":
        return group.energy, group.mu, _AXIS_LABELS["mu"]
    elif plot_type == "norm":
        return group.energy, group.norm, _AXIS_LABELS["norm"]
    elif plot_type == "flat":
        return group.energy, group.flat, _AXIS_LABELS["flat"]
    elif plot_type == "dmude":
        return group.energy, group.dmude, _AXIS_LABELS["dmude"]
    elif plot_type == "chi_k":
        x = group.k
        y = group.chi * group.k ** kweight
        return x, y, (r"k ($\AA^{-1}$)", _chi_ylabel(kweight))
    elif plot_type == "chi_r":
        return (
            group.r,
            group.chir_mag,
            (r"R ($\AA$)", _chir_ylabel(kweight)),
        )
    elif plot_type == "chi_r_mag":
        return (
            group.r,
            group.chir_mag,
            (r"R ($\AA$)", _chir_ylabel(kweight)),
        )
    elif plot_type == "chi_q":
        return (
            group.q,
            group.chiq_mag,
            (r"q ($\AA^{-1}$)", _chiq_ylabel(kweight)),
        )
    elif plot_type == "cauchy_wavelet":
        # 2D data — handled specially in the plot function
        raise ValueError("cauchy_wavelet is a 2D plot; use the dedicated branch in larch_plot.")
    else:
        raise ValueError(
            f"Unknown plot_type '{plot_type}'. "
            "Valid types: mu, norm, flat, dmude, chi_k, chi_r, chi_r_mag, chi_q, cauchy_wavelet"
        )


def register(mcp: FastMCP) -> None:
    @mcp.tool(name="larch_plot")
    def larch_plot(
        ctx: Context,
        group_id: str,
        plot_type: str = "norm",
        kweight: int = 2,
        xmin: float | None = None,
        xmax: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
        title: str | None = None,
        overlay_group_ids: list[str] | None = None,
        show_e0: bool = False,
        show_pre_post_edge: bool = False,
        figsize_w: float = 8.0,
        figsize_h: float = 5.0,
        dpi: int = 150,
        save_path: str | None = None,
    ) -> dict:
        """Plot spectral data with publication-quality formatting.

        Args:
            group_id: Primary group to plot.
            plot_type: Type of plot. One of:
                "mu" - raw mu(E)
                "norm" - normalized mu(E)
                "flat" - flattened mu(E)
                "dmude" - derivative dmu/dE
                "chi_k" - chi(k) weighted by k^kweight
                "chi_r" or "chi_r_mag" - |chi(R)| magnitude
                "chi_q" - |chi(q)| back-transform magnitude
                "cauchy_wavelet" - 2D wavelet transform contour (k vs R)
            kweight: k-weighting for chi plots (0, 1, 2, or 3). Default 2.
            xmin: X-axis minimum.
            xmax: X-axis maximum.
            ymin: Y-axis minimum.
            ymax: Y-axis maximum.
            title: Plot title. Auto-generated if not provided.
            overlay_group_ids: List of additional group IDs to overlay.
            show_e0: Show vertical line at E0 (for energy-space plots).
            show_pre_post_edge: Show pre-edge and post-edge fit lines (mu plot only).
            figsize_w: Figure width in inches. Default 8.
            figsize_h: Figure height in inches. Default 5.
            dpi: Resolution in dots per inch. Default 150.
            save_path: If provided, save PNG to this path instead of returning base64.

        Returns:
            Base64-encoded PNG image data, or filepath if save_path was given.
        """
        session = _get_session(ctx)
        try:
            group = session.get_group(group_id)
        except KeyError as e:
            return {"error": str(e)}

        # Validate required data exists
        required = {
            "mu": ["energy", "mu"],
            "norm": ["energy", "norm"],
            "flat": ["energy", "flat"],
            "dmude": ["energy", "dmude"],
            "chi_k": ["k", "chi"],
            "chi_r": ["r", "chir_mag"],
            "chi_r_mag": ["r", "chir_mag"],
            "chi_q": ["q", "chiq_mag"],
            "cauchy_wavelet": ["k", "wcauchy_mag", "wcauchy_r"],
        }
        if plot_type not in required:
            return {
                "error": f"Unknown plot_type '{plot_type}'. Valid: {list(required.keys())}"
            }
        for attr in required[plot_type]:
            if not hasattr(group, attr):
                return {
                    "error": f"Group '{group_id}' missing '{attr}'. Required processing may not have been run."
                }

        try:
            # --- 2D wavelet contour plot ---
            if plot_type == "cauchy_wavelet":
                fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
                im = ax.contourf(
                    group.k, group.wcauchy_r, group.wcauchy_mag,
                    levels=100, cmap="jet",
                )
                ax.set_xlabel(r"k ($\AA^{-1}$)", fontsize=12)
                ax.set_ylabel(r"R ($\AA$)", fontsize=12)
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label("|WT|")

                if xmin is not None or xmax is not None:
                    ax.set_xlim(xmin, xmax)
                if ymin is not None or ymax is not None:
                    ax.set_ylim(ymin, ymax)

                wt_title = title or f"Cauchy Wavelet Transform — {group_id}"
                ax.set_title(wt_title, fontsize=13)
                ax.tick_params(labelsize=10)
                fig.tight_layout()

                if save_path:
                    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
                    plt.close(fig)
                    return {"saved_to": save_path, "dpi": dpi}
                else:
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
                    plt.close(fig)
                    buf.seek(0)
                    b64 = base64.b64encode(buf.read()).decode()
                    return {
                        "image_base64": b64,
                        "format": "png",
                        "dpi": dpi,
                        "plot_type": plot_type,
                        "group_id": group_id,
                    }

            fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))

            # Plot primary group
            x, y, (xlabel, ylabel) = _get_plot_data(group, plot_type, kweight)
            ax.plot(x, y, label=group_id, linewidth=1.5)

            # Show E0 line
            if show_e0 and hasattr(group, "e0") and plot_type in (
                "mu",
                "norm",
                "flat",
                "dmude",
            ):
                ax.axvline(
                    group.e0,
                    color="gray",
                    linestyle="--",
                    alpha=0.7,
                    label=f"E0={group.e0:.1f} eV",
                )

            # Show pre/post edge lines
            if (
                show_pre_post_edge
                and plot_type == "mu"
                and hasattr(group, "pre_edge")
                and hasattr(group, "post_edge")
            ):
                ax.plot(
                    group.energy,
                    group.pre_edge,
                    "--",
                    color="C2",
                    alpha=0.7,
                    label="pre-edge",
                )
                ax.plot(
                    group.energy,
                    group.post_edge,
                    "--",
                    color="C3",
                    alpha=0.7,
                    label="post-edge",
                )

            # Overlay additional groups
            if overlay_group_ids:
                for oid in overlay_group_ids:
                    try:
                        ogroup = session.get_group(oid)
                        ox, oy, _ = _get_plot_data(ogroup, plot_type, kweight)
                        ax.plot(ox, oy, label=oid, linewidth=1.5)
                    except (KeyError, AttributeError, ValueError) as oe:
                        # Skip groups that can't be plotted
                        ax.text(
                            0.5,
                            0.02,
                            f"Could not overlay '{oid}': {oe}",
                            transform=ax.transAxes,
                            fontsize=8,
                            ha="center",
                            color="red",
                        )

            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)

            if xmin is not None or xmax is not None:
                ax.set_xlim(xmin, xmax)
            if ymin is not None or ymax is not None:
                ax.set_ylim(ymin, ymax)

            if title:
                ax.set_title(title, fontsize=13)
            else:
                type_names = {
                    "mu": "Raw Absorption",
                    "norm": "Normalized XANES",
                    "flat": "Flattened XANES",
                    "dmude": "Derivative",
                    "chi_k": "EXAFS",
                    "chi_r": "Fourier Transform Magnitude",
                    "chi_r_mag": "Fourier Transform Magnitude",
                    "chi_q": "Back Transform",
                    "cauchy_wavelet": "Cauchy Wavelet Transform",
                }
                ax.set_title(
                    f"{type_names.get(plot_type, plot_type)} — {group_id}",
                    fontsize=13,
                )

            if overlay_group_ids or show_e0 or show_pre_post_edge:
                ax.legend(fontsize=10)

            ax.tick_params(labelsize=10)
            fig.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                return {"saved_to": save_path, "dpi": dpi}
            else:
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode()
                return {
                    "image_base64": b64,
                    "format": "png",
                    "dpi": dpi,
                    "plot_type": plot_type,
                    "group_id": group_id,
                }

        except Exception as e:
            plt.close("all")
            return {"error": f"Plotting error: {e}"}
