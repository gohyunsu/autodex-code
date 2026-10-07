#!/usr/bin/env python3
"""Write complete machine- and human-readable presentation asset indexes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = (
    Path.home() / "shared_data" / "AutoDex" / "precision_insertion" /
    "presentation_assets"
)


def _category(relative: Path) -> str:
    if relative.suffix.lower() == ".gif":
        return "animation_gif"
    if relative.suffix.lower() == ".mp4":
        return "animation_mp4"
    if relative.suffix.lower() == ".png":
        return "still_image"
    if "mesh_cache" in relative.parts or "blender_original_robot_visuals" in relative.parts:
        return "render_cache"
    if relative.suffix.lower() in {".npz", ".npy", ".stl", ".ply"}:
        return "regeneration_data"
    if relative.suffix.lower() == ".json":
        return "evidence_or_manifest"
    if relative.suffix.lower() == ".md":
        return "documentation"
    return "other"


def _status_summary(path: Path, root: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path.relative_to(root)),
        "status": payload.get("status"),
        "active_video": payload.get("active_video"),
        "active_gif": payload.get("active_gif"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    inventory_path = root / "asset_inventory.json"
    index_path = root / "ASSET_INDEX.md"
    excluded_names = {inventory_path.name, index_path.name}
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and "audit" not in path.relative_to(root).parts
        and path.name not in excluded_names
    )
    records = [{
        "path": str(path.relative_to(root)),
        "absolute_path": str(path),
        "category": _category(path.relative_to(root)),
        "size_bytes": path.stat().st_size,
    } for path in files]
    counts: dict[str, int] = {}
    sizes: dict[str, int] = {}
    for record in records:
        category = record["category"]
        counts[category] = counts.get(category, 0) + 1
        sizes[category] = sizes.get(category, 0) + record["size_bytes"]
    statuses = [
        _status_summary(path, root)
        for path in sorted(root.glob("*/status.json"))
    ]
    payload = {
        "schema_version": 1,
        "status": "complete_active_presentation_asset_inventory",
        "root": str(root),
        "excluded": ["all paths under audit/", "this inventory and its Markdown view"],
        "file_count": len(records),
        "counts_by_category": counts,
        "bytes_by_category": sizes,
        "status_files": statuses,
        "files": records,
    }
    inventory_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    gif_manifest = json.loads((root / "gif_manifest.json").read_text(encoding="utf-8"))
    stills = [r for r in records if r["category"] == "still_image"]
    evidence = [
        r for r in records
        if r["category"] == "evidence_or_manifest"
        and (r["path"].endswith("status.json") or r["path"] in {
            "manifest.json", "gif_manifest.json"
        })
    ]
    lines = [
        "# Precision insertion asset index",
        "",
        f"Root: `{root}`",
        "",
        "The MP4 files are archival/high-quality renders. Every active MP4 has a "
        "960×540, 10 fps GIF beside it. Paths under `audit/` are intentionally "
        "excluded from the active set.",
        "",
        "## Authoritative indexes and status",
        "",
        f"- `{root / 'manifest.json'}`",
        f"- `{root / 'gif_manifest.json'}`",
        f"- `{inventory_path}`",
    ]
    for record in evidence:
        if record["path"] not in {"manifest.json", "gif_manifest.json"}:
            lines.append(f"- `{root / record['path']}`")
    lines.extend(["", "## Animation pairs", ""])
    for record in gif_manifest["assets"]:
        lines.append(f"- `{root / record['source_mp4']}`")
        lines.append(f"  - GIF: `{root / record['gif']}`")
    lines.extend(["", "## Still images", ""])
    for record in stills:
        lines.append(f"- `{root / record['path']}`")
    lines.extend([
        "", "## Regeneration and evidence files", "",
        "The full list, including NPZ trajectory bundles, Blender bundles, JSON "
        "reports, and render caches, is in `asset_inventory.json`. Render-cache "
        "PLY files are indexed there but are not slide deliverables.", "",
        "## Evidence boundary", "",
        "- Reorientation: continuous AutoDex runtime cuRobo motion is available; "
        "the 12 cm drop, post-drop classification, and robot execution are not validated.",
        "- Optional finish: exact-mesh geometry only; no press pose, force limits, "
        "controller, or robot plan is claimed.",
        "- Insertion candidate videos remain offline previews unless their per-asset "
        "report states otherwise.", "",
    ])
    index_path.write_text("\n".join(lines), encoding="utf-8")
    print(inventory_path)
    print(index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
