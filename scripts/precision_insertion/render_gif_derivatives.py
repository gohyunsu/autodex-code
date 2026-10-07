#!/usr/bin/env python3
"""Create presentation-friendly GIF derivatives for every active MP4 asset.

By default the script scans the precision-insertion presentation tree, skips
all ``audit`` directories, and writes ``video.gif`` beside each ``video.mp4``.
The source MP4 remains the archival/high-quality version.  GIFs are capped at
960 px width and 10 fps because full-resolution 1280x720/20 fps GIFs are
unnecessarily large for slides and GitHub previews.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


DEFAULT_ROOT = (
    Path.home() / "shared_data" / "AutoDex" / "precision_insertion" /
    "presentation_assets"
)


def _probe(path: Path) -> dict[str, Any]:
    process = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
            "-show_entries", "format=duration,size", "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(process.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--colors", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--include-audit",
        action="store_true",
        help="also convert deprecated/audit MP4 files; they remain non-active evidence",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required")
    if args.fps <= 0 or args.width <= 0 or not 2 <= args.colors <= 256:
        parser.error("fps/width must be positive and colors must be in [2, 256]")

    videos = sorted(
        path for path in root.rglob("*.mp4")
        if args.include_audit or "audit" not in path.relative_to(root).parts
    )
    records: list[dict[str, Any]] = []
    for index, source in enumerate(videos, start=1):
        output = source.with_suffix(".gif")
        action = "kept"
        if args.force or not output.is_file() or output.stat().st_mtime < source.stat().st_mtime:
            action = "rendered"
            filter_graph = (
                f"fps={args.fps},scale={args.width}:-1:flags=lanczos,"
                "split[source][palette_source];"
                f"[palette_source]palettegen=max_colors={args.colors}:"
                "stats_mode=diff[palette];"
                "[source][palette]paletteuse=dither=bayer:bayer_scale=4:"
                "diff_mode=rectangle"
            )
            subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-y", "-i", str(source),
                    "-filter_complex", filter_graph, "-loop", "0", str(output),
                ],
                check=True,
            )
        source_probe = _probe(source)
        gif_probe = _probe(output)
        record = {
            "source_mp4": str(source.relative_to(root)),
            "gif": str(output.relative_to(root)),
            "action": action,
            "source": source_probe,
            "derivative": gif_probe,
            "source_size_bytes": source.stat().st_size,
            "gif_size_bytes": output.stat().st_size,
            "deprecated_audit": "audit" in source.relative_to(root).parts,
        }
        records.append(record)
        print(f"[{index}/{len(videos)}] {action}: {record['gif']}")

    manifest = {
        "schema_version": 1,
        "status": (
            "all_mp4_assets_including_deprecated_audit_have_gif_derivatives"
            if args.include_audit
            else "all_active_mp4_assets_have_gif_derivatives"
        ),
        "root": str(root),
        "includes_deprecated_audit": args.include_audit,
        "excludes": ([] if args.include_audit else ["any path component named audit"]),
        "settings": {
            "fps": args.fps,
            "maximum_width_px": args.width,
            "palette_colors": args.colors,
            "loop": "infinite",
        },
        "count": len(records),
        "assets": records,
    }
    manifest_path = root / "gif_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
