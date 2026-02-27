"""Larch session and group state manager."""

from __future__ import annotations

from typing import Any

import numpy as np
from larch import Group

from .util import make_group_id, summarize_group, unique_group_id


class SessionManager:
    """Manages a persistent collection of named larch Groups.

    Groups are stored in a dict keyed by user-friendly IDs
    (auto-generated from filenames or user-supplied).
    """

    def __init__(self) -> None:
        self._groups: dict[str, Group] = {}

    def add_group(
        self,
        group: Group,
        group_id: str | None = None,
        filepath: str | None = None,
        label: str | None = None,
    ) -> str:
        """Add a group and return its assigned ID."""
        base_id = group_id or make_group_id(filepath=filepath, label=label)
        gid = unique_group_id(base_id, set(self._groups.keys()))
        self._groups[gid] = group
        return gid

    def get_group(self, group_id: str) -> Group:
        """Get a group by ID. Raises KeyError if not found."""
        if group_id not in self._groups:
            available = ", ".join(sorted(self._groups.keys())) or "(none)"
            raise KeyError(
                f"Group '{group_id}' not found. Available groups: {available}"
            )
        return self._groups[group_id]

    def remove_group(self, group_id: str) -> None:
        """Remove a group by ID."""
        if group_id not in self._groups:
            available = ", ".join(sorted(self._groups.keys())) or "(none)"
            raise KeyError(
                f"Group '{group_id}' not found. Available groups: {available}"
            )
        del self._groups[group_id]

    def list_groups(self) -> list[dict]:
        """List all groups with summary info."""
        result = []
        for gid, group in self._groups.items():
            summary = summarize_group(group, gid)
            # Return a compact version for listing
            result.append(
                {
                    "group_id": gid,
                    "arrays": list(summary["arrays"].keys()),
                    "n_points": (
                        summary["arrays"].get("energy", {}).get("shape", [None])[0]
                        or summary["arrays"]
                        .get("mu", {})
                        .get("shape", [None])[0]
                    ),
                    "processing": summary["processing"],
                    "e0": summary["scalars"].get("e0"),
                    "edge_step": summary["scalars"].get("edge_step"),
                }
            )
        return result

    def inspect_group(self, group_id: str) -> dict:
        """Return detailed attribute listing for a group."""
        group = self.get_group(group_id)
        return summarize_group(group, group_id)

    @property
    def group_ids(self) -> list[str]:
        return list(self._groups.keys())
