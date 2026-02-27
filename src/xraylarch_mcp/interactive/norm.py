"""Interactive normalization parameter selector with matplotlib sliders.

Standalone script invoked as a subprocess by the MCP tool.
Uses TkAgg backend for GUI display.

Usage:
    python -m xraylarch_mcp.interactive.norm <data_npz> <output_json> [options]

Options:
    --e0 VAL        Edge energy in eV (auto-detected if omitted)
    --pre1 VAL      Pre-edge fit lower bound relative to E0 (default: -150)
    --pre2 VAL      Pre-edge fit upper bound relative to E0 (default: -30)
    --norm1 VAL     Post-edge fit lower bound relative to E0 (default: 100)
    --norm2 VAL     Post-edge fit upper bound relative to E0 (default: 500)
    --nnorm VAL     Post-edge polynomial degree 1-3 (default: 2)
    --label NAME    Label for plot title
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

# Force TkAgg before any other matplotlib import
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from larch import Group
from larch.xafs import pre_edge


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Interactive XANES normalization")
    p.add_argument("data_npz", help="Path to .npz file with energy and mu arrays")
    p.add_argument("output_json", help="Path to write final parameters as JSON")
    p.add_argument("--e0", type=float, default=None)
    p.add_argument("--pre1", type=float, default=-150.0)
    p.add_argument("--pre2", type=float, default=-30.0)
    p.add_argument("--norm1", type=float, default=100.0)
    p.add_argument("--norm2", type=float, default=500.0)
    p.add_argument("--nnorm", type=int, default=2)
    p.add_argument("--label", type=str, default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # Load data from npz
    data = np.load(args.data_npz)
    energy = data["energy"]
    mu = data["mu"]

    # Build a larch Group
    group = Group(energy=energy, mu=mu)

    # Auto-detect E0 if not provided
    if args.e0 is None:
        pre_edge(group)
        e0 = float(group.e0)
    else:
        e0 = args.e0

    # Compute slider ranges from data
    e_range = float(energy.max() - energy.min())
    e_min_rel = float(energy.min() - e0)
    e_max_rel = float(energy.max() - e0)

    # Slider bounds (clamped to data range)
    pre1_lo = max(e_min_rel, -e_range)
    pre1_hi = min(-5.0, e_max_rel)
    pre2_lo = max(e_min_rel, -200.0)
    pre2_hi = min(-2.0, e_max_rel)
    norm1_lo = max(5.0, 5.0)
    norm1_hi = min(e_max_rel, 600.0)
    norm2_lo = max(50.0, 50.0)
    norm2_hi = min(e_max_rel, e_max_rel)

    # Clamp initial values to valid ranges
    init_pre1 = float(np.clip(args.pre1, pre1_lo, pre1_hi))
    init_pre2 = float(np.clip(args.pre2, pre2_lo, pre2_hi))
    init_norm1 = float(np.clip(args.norm1, norm1_lo, norm1_hi))
    init_norm2 = float(np.clip(args.norm2, norm2_lo, norm2_hi))
    init_nnorm = int(np.clip(args.nnorm, 1, 3))

    # Track whether user explicitly saved
    saved = [False]

    # --- Build figure ---
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(bottom=0.38)

    label_text = args.label or "Spectrum"
    fig.suptitle(f"Interactive Normalization: {label_text}", fontsize=13)

    ax_mu = fig.add_subplot(121)
    ax_norm = fig.add_subplot(122)

    # Slider axes
    slider_color = "lightgoldenrodyellow"
    ax_pre1 = fig.add_axes([0.12, 0.24, 0.75, 0.03], facecolor=slider_color)
    ax_pre2 = fig.add_axes([0.12, 0.19, 0.75, 0.03], facecolor=slider_color)
    ax_norm1 = fig.add_axes([0.12, 0.14, 0.75, 0.03], facecolor=slider_color)
    ax_norm2 = fig.add_axes([0.12, 0.09, 0.75, 0.03], facecolor=slider_color)
    ax_nnorm = fig.add_axes([0.12, 0.04, 0.75, 0.03], facecolor=slider_color)

    s_pre1 = Slider(ax_pre1, "pre1", pre1_lo, pre1_hi,
                     valinit=init_pre1, valstep=5)
    s_pre2 = Slider(ax_pre2, "pre2", pre2_lo, pre2_hi,
                     valinit=init_pre2, valstep=5)
    s_norm1 = Slider(ax_norm1, "norm1", norm1_lo, norm1_hi,
                      valinit=init_norm1, valstep=5)
    s_norm2 = Slider(ax_norm2, "norm2", norm2_lo, norm2_hi,
                      valinit=init_norm2, valstep=10)
    s_nnorm = Slider(ax_nnorm, "nnorm", 1, 3,
                      valinit=init_nnorm, valstep=1)

    def get_params() -> dict:
        return {
            "e0": e0,
            "pre1": float(s_pre1.val),
            "pre2": float(s_pre2.val),
            "norm1": float(s_norm1.val),
            "norm2": float(s_norm2.val),
            "nnorm": int(s_nnorm.val),
        }

    def update_plot(val=None):
        ax_mu.clear()
        ax_norm.clear()

        p = get_params()
        try:
            pre_edge(group, e0=e0,
                     pre1=p["pre1"], pre2=p["pre2"],
                     norm1=p["norm1"], norm2=p["norm2"],
                     nnorm=p["nnorm"])
        except Exception as exc:
            ax_mu.text(0.5, 0.5, f"Error: {exc}",
                       transform=ax_mu.transAxes, ha="center", color="red")
            fig.canvas.draw_idle()
            return

        # Left panel: raw mu with pre/post edge lines
        ax_mu.plot(energy, group.mu, "b-", lw=1.5, label=r"$\mu$(E)")
        ax_mu.plot(energy, group.pre_edge, "--", color="green", lw=1.2,
                   label="pre-edge")
        ax_mu.plot(energy, group.post_edge, "--", color="red", lw=1.2,
                   label="post-edge")
        ax_mu.axvline(e0, color="gray", ls="--", alpha=0.6)

        # Shade fit regions
        ax_mu.axvspan(e0 + p["pre1"], e0 + p["pre2"],
                      alpha=0.1, color="green", label="pre-edge range")
        ax_mu.axvspan(e0 + p["norm1"], e0 + p["norm2"],
                      alpha=0.1, color="red", label="post-edge range")

        ax_mu.set_xlabel("Energy (eV)", fontsize=11)
        ax_mu.set_ylabel(r"$\mu$(E)", fontsize=11)
        ax_mu.set_title("Raw Absorption", fontsize=12)
        ax_mu.legend(fontsize=8, loc="best")

        # Right panel: flattened normalized
        ax_norm.plot(energy, group.flat, "b-", lw=1.5)
        ax_norm.axhline(1.0, color="gray", ls=":", alpha=0.5)
        ax_norm.axhline(0.0, color="gray", ls=":", alpha=0.5)
        ax_norm.axvline(e0, color="gray", ls="--", alpha=0.6)
        ax_norm.set_xlabel("Energy (eV)", fontsize=11)
        ax_norm.set_ylabel(r"Flattened $\mu$(E)", fontsize=11)
        ax_norm.set_title(
            f"Flattened (edge_step={group.edge_step:.4f})", fontsize=12)

        fig.canvas.draw_idle()

    for s in [s_pre1, s_pre2, s_norm1, s_norm2, s_nnorm]:
        s.on_changed(update_plot)

    # Save button
    def save_callback(event):
        p = get_params()
        p["edge_step"] = float(group.edge_step)
        with open(args.output_json, "w") as f:
            json.dump(p, f, indent=2)
        saved[0] = True
        print(f"Parameters saved to: {args.output_json}")

    ax_save = fig.add_axes([0.80, 0.28, 0.08, 0.04])
    btn_save = Button(ax_save, "Save", color="lightblue",
                      hovercolor="deepskyblue")
    btn_save.on_clicked(save_callback)

    # Initial draw
    update_plot()
    plt.show()

    # On window close, always write final params
    final = get_params()
    final["edge_step"] = float(group.edge_step)

    with open(args.output_json, "w") as f:
        json.dump(final, f, indent=2)

    # Print to stdout for the calling process
    print(json.dumps(final))


if __name__ == "__main__":
    main()
