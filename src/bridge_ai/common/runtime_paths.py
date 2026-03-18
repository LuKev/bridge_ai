"""Workspace-aware path helpers for Bazel and local runs."""

from __future__ import annotations

import os
from pathlib import Path


def workspace_root() -> Path:
    root = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if root:
        return Path(root)
    return Path.cwd()


def resolve_runtime_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return workspace_root() / candidate
