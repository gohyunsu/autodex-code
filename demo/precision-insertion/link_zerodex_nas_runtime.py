#!/usr/bin/env python3
"""Create a compact, path-correct runtime tree inside the NAS project.

Heavy immutable directories are linked once. Small scene/configuration JSON
files are copied with rebased paths. The old AutoDex tree is never consulted.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


LINKED_DIRECTORIES = frozenset({
    Path("AutoDex/candidates"),
    Path("AutoDex/foundpose_assets"),
    Path("AutoDex/models"),
    Path("AutoDex/weights"),
    Path("AutoDex/precision_insertion/zerodex_capture4/calibration"),
    Path("AutoDex/precision_insertion/zerodex_capture4/robot_deps"),
})


def link_runtime(project: Path) -> dict[str, object]:
    project = project.expanduser().resolve()
    manifest = json.loads((project / "PROJECT_MANIFEST.json").read_text())
    source = project / "assets/shared_data"
    target = project / "runtime/active_shared_data"
    target.mkdir(parents=True, exist_ok=True)
    original = manifest["source_shared_root"]
    linked = copied = linked_dirs = 0
    for root, dirs, files in os.walk(source):
        relative = Path(root).relative_to(source)
        out = target / relative
        out.mkdir(parents=True, exist_ok=True)
        for dirname in list(dirs):
            child = relative / dirname
            dst_dir = target / child
            if child in LINKED_DIRECTORIES:
                src_dir = source / child
                if dst_dir.is_symlink() and dst_dir.readlink() == src_dir:
                    pass
                elif dst_dir.exists() or dst_dir.is_symlink():
                    raise FileExistsError(f"runtime directory needs a link: {dst_dir}")
                else:
                    dst_dir.symlink_to(src_dir, target_is_directory=True)
                dirs.remove(dirname)
                linked_dirs += 1
            else:
                dst_dir.mkdir(exist_ok=True)
        for filename in files:
            src = Path(root) / filename
            dst = out / filename
            if src.suffix == ".json":
                before = src.read_text(encoding="utf-8")
                after = before.replace(original, str(target)).replace(
                    "/home/robot/shared_data", str(target))
                if not dst.is_file() or dst.read_text(encoding="utf-8") != after:
                    dst.write_text(after, encoding="utf-8")
                copied += 1
            else:
                if dst.is_symlink() and dst.readlink() == src:
                    pass
                else:
                    dst.unlink(missing_ok=True)
                    dst.symlink_to(src)
                linked += 1
    (target / "AutoDex/precision_insertion/zerodex_capture4/runs").mkdir(
        parents=True, exist_ok=True)
    return {"source": str(source), "runtime": str(target),
            "linked_directories": linked_dirs, "linked_files": linked,
            "rewritten_json_files": copied}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path,
                        default=Path("/mnt/paradex2/hyunsu/precision_insertion"))
    args = parser.parse_args()
    print(json.dumps(link_runtime(args.project), indent=2))


if __name__ == "__main__":
    main()
