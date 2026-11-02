"""Machine-dependent roots for the independent precision-insertion project.

Runtime launchers normally export these variables from the selected machine
profile. Standalone maintenance tools and the capture4 adapter use the same
names here instead of embedding another copy of the workstation layout.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator, Mapping


DEFAULT_NAS_PROJECT_ROOT = Path(
    "/mnt/paradex2/hyunsu/precision_insertion")
DEFAULT_CAPTURE4_PROJECT_ROOT = Path(
    "/home/capture4/shared_data/hyunsu/precision_insertion")

_RUNTIME_PATH_VARIABLES = (
    "PRECISION_NAS_PROJECT_ROOT",
    "PRECISION_SHARED_ROOT",
    "PRECISION_STAGE_ROOT",
    "PRECISION_RUN_OUTPUT_ROOT",
    "PRECISION_RECORDING_RECEIPT_ROOT",
    "PRECISION_DEV_ROOT",
    "PRECISION_ENDPOINT_CATALOG",
    "PRECISION_ROBOT_CONFIG",
    "PRECISION_SAM_CHECKPOINT",
    "PRECISION_LOCAL_PROJECT_ROOT",
    "PRECISION_TIMING_LOCAL_ROOT",
    "PRECISION_PARADEX_ROOT",
    "PRECISION_FR3_INSPIRE_URDF",
)


def _classify_path(path: Path, *, allowed: Path, mount: Path) -> dict:
    resolved = path.expanduser().resolve(strict=False)
    on_nas = resolved == mount or mount in resolved.parents
    within_allowed = (
        not on_nas or resolved == allowed or allowed in resolved.parents)
    return {
        "path": str(resolved),
        "storage": "nas" if on_nas else "local",
        "within_allowed_nas_root": within_allowed,
    }


def _absolute_json_strings(value, field: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_field = f"{field}.{key}" if field else str(key)
            yield from _absolute_json_strings(child, child_field)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _absolute_json_strings(child, f"{field}[{index}]")
    elif isinstance(value, str) and value.startswith("/"):
        yield field, value


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


def runtime_namespace_report(
    environ: Mapping[str, str] | None = None,
) -> dict | None:
    """Report active paths and any ParaDex2 path outside Hyunsu's root.

    The shell launcher enforces the namespace before execution. Recording the
    same resolved paths in static preflight makes that separation auditable.
    """
    values = os.environ if environ is None else environ
    allowed_value = values.get("PRECISION_ALLOWED_NAS_ROOT")
    if not allowed_value:
        return None
    allowed = Path(allowed_value).expanduser().resolve(strict=False)
    mount = Path(values.get(
        "PRECISION_NAS_MOUNT_ROOT", "/mnt/paradex2"
    )).expanduser().resolve(strict=False)
    entries, violations = [], []
    for name in _RUNTIME_PATH_VARIABLES:
        raw = values.get(name)
        if not raw:
            continue
        row = {"variable": name, **_classify_path(
            Path(raw), allowed=allowed, mount=mount)}
        entries.append(row)
        if not row["within_allowed_nas_root"]:
            violations.append(row)
    return {
        "schema": "precision_insertion_runtime_namespace_v1",
        "nas_mount_root": str(mount),
        "allowed_nas_root": str(allowed),
        "paths": entries,
        "violations": violations,
        "all_nas_paths_within_allowed_root": not violations,
    }


def runtime_json_namespace_report(
    sources: Mapping[str, str | Path],
    environ: Mapping[str, str] | None = None,
) -> dict | None:
    """Audit every absolute path embedded in active runtime JSON inputs.

    Valid local paths remain allowed. Only paths resolving below the ParaDex2
    mount but outside the configured user namespace are violations.
    """
    values = os.environ if environ is None else environ
    allowed_value = values.get("PRECISION_ALLOWED_NAS_ROOT")
    if not allowed_value:
        return None
    allowed = Path(allowed_value).expanduser().resolve(strict=False)
    mount = Path(values.get(
        "PRECISION_NAS_MOUNT_ROOT", "/mnt/paradex2"
    )).expanduser().resolve(strict=False)
    references = 0
    unique: dict[str, dict] = {}
    violations: list[dict] = []
    source_counts: dict[str, int] = {}
    for source_name, source_value in sources.items():
        source = Path(source_value).expanduser().resolve(strict=False)
        source_row = _classify_path(source, allowed=allowed, mount=mount)
        if not source_row["within_allowed_nas_root"]:
            violations.append({
                "source": source_name,
                "source_path": str(source),
                "field": "$source",
                **source_row,
            })
            source_counts[source_name] = 0
            continue
        record = json.loads(source.read_text(encoding="utf-8"))
        count = 0
        for field, raw in _absolute_json_strings(record):
            count += 1
            references += 1
            row = _classify_path(Path(raw), allowed=allowed, mount=mount)
            unique.setdefault(row["path"], row)
            if not row["within_allowed_nas_root"]:
                violations.append({
                    "source": source_name,
                    "source_path": str(source),
                    "field": field,
                    **row,
                })
        source_counts[source_name] = count
    return {
        "schema": "precision_insertion_runtime_json_namespace_v1",
        "nas_mount_root": str(mount),
        "allowed_nas_root": str(allowed),
        "sources": {name: str(Path(path).expanduser().resolve(strict=False))
                    for name, path in sources.items()},
        "absolute_path_references": references,
        "unique_absolute_paths": len(unique),
        "references_by_source": source_counts,
        "violations": violations,
        "all_nas_paths_within_allowed_root": not violations,
    }
