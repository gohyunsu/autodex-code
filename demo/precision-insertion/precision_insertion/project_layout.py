"""Machine-dependent roots for the independent precision-insertion project.

Runtime launchers normally export these variables from the selected machine
profile. Standalone maintenance tools and the capture4 adapter use the same
names here instead of embedding another copy of the workstation layout.
"""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_NAS_PROJECT_ROOT = Path(
    "/mnt/paradex2/hyunsu/precision_insertion")
DEFAULT_CAPTURE4_PROJECT_ROOT = Path(
    "/home/capture4/shared_data/hyunsu/precision_insertion")


def nas_project_root(value: str | Path | None = None) -> Path:
    """Return the configured main-PC/NAS project root."""
    selected = (value if value is not None else
                os.environ.get("PRECISION_NAS_PROJECT_ROOT",
                               str(DEFAULT_NAS_PROJECT_ROOT)))
    return Path(selected).expanduser()


def capture4_project_root(value: str | Path | None = None) -> Path:
    """Return capture4's view of the same project namespace."""
    selected = (value if value is not None else
                os.environ.get("PRECISION_CAPTURE4_PROJECT_ROOT",
                               str(DEFAULT_CAPTURE4_PROJECT_ROOT)))
    return Path(selected).expanduser()


def map_main_project_path_to_capture4(
    source: str | Path,
    *,
    main_root: str | Path | None = None,
    capture_root: str | Path | None = None,
) -> Path:
    """Map one project path between PCs; leave unrelated paths untouched."""
    path = Path(source).expanduser()
    main = nas_project_root(main_root)
    capture = capture4_project_root(capture_root)
    try:
        relative = path.relative_to(main)
    except ValueError:
        return path
    return capture / relative
