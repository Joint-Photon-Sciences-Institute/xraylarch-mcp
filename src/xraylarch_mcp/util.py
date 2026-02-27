"""Shared helpers: error formatting, group serialization."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np


def make_group_id(filepath: str | None = None, label: str | None = None) -> str:
    """Generate a group ID from a filepath or label.

    Converts to lowercase, replaces non-alphanumeric chars with underscores,
    strips leading/trailing underscores.
    """
    if label:
        name = label
    elif filepath:
        name = Path(filepath).stem
    else:
        name = "group"
    # Normalize: lowercase, replace non-alnum with underscore
    name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return name or "group"


def unique_group_id(base_id: str, existing: set[str]) -> str:
    """Ensure a group ID is unique by appending _1, _2, etc."""
    if base_id not in existing:
        return base_id
    i = 1
    while f"{base_id}_{i}" in existing:
        i += 1
    return f"{base_id}_{i}"


def summarize_group(group: Any, group_id: str) -> dict:
    """Summarize a larch Group's attributes for tool responses."""
    arrays = {}
    scalars = {}
    processing = []
    other = {}

    # Track what processing has been done based on known attribute markers
    processing_markers = {
        "norm": "pre_edge",
        "chi": "autobk",
        "chir_mag": "xftf",
        "chiq_mag": "xftr",
        "epsilon_k": "estimate_noise",
    }

    for attr in sorted(dir(group)):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(group, attr)
        except Exception:
            continue

        if isinstance(val, np.ndarray):
            info: dict[str, Any] = {
                "shape": list(val.shape),
                "min": float(np.nanmin(val)) if val.size > 0 else None,
                "max": float(np.nanmax(val)) if val.size > 0 else None,
            }
            # Add units for known arrays
            units = {
                "energy": "eV",
                "k": "\u00c5\u207b\u00b9",
                "r": "\u00c5",
                "q": "\u00c5\u207b\u00b9",
            }
            if attr in units:
                info["unit"] = units[attr]
            arrays[attr] = info
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            scalars[attr] = round(val, 6) if isinstance(val, float) else val
        elif isinstance(val, str):
            scalars[attr] = val
        # Skip callables, Group objects, etc. for summary

        if attr in processing_markers:
            processing.append(processing_markers[attr])

    return {
        "group_id": group_id,
        "arrays": arrays,
        "scalars": scalars,
        "processing": processing,
    }


def format_error(operation: str, error: Exception, hints: str | None = None) -> str:
    """Format an error message with optional hints."""
    msg = f"Error in {operation}: {type(error).__name__}: {error}"
    if hints:
        msg += f"\nHint: {hints}"
    return msg
