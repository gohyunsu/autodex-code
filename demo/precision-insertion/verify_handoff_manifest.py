#!/usr/bin/env python3
"""Read back a NAS handoff's declared bytes before recipient installation.

Supports the historical cylinder candidate MANIFEST and generated FoundPose
archive manifest. It checks integrity, not mesh identity or pose accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path) -> dict:
    bundle = Path(root).expanduser().resolve()
    candidates = [bundle / "MANIFEST.json", bundle / "manifest.json"]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise ValueError("handoff needs exactly one recognized manifest")
    manifest_path = found[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("handoff manifest must be an object")
    hashes = manifest.get("file_sha256")
    if isinstance(hashes, dict) and hashes:
        declared = hashes
    elif (isinstance(manifest.get("archive"), str) and
          isinstance(manifest.get("archive_sha256_read_back"), str)):
        declared = {manifest["archive"]: manifest["archive_sha256_read_back"]}
    else:
        raise ValueError("unsupported handoff manifest digest format")
    total_bytes = 0
    for relative, digest in declared.items():
        if (not isinstance(relative, str) or not relative or
                not isinstance(digest, str) or len(digest) != 64 or
                any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError("invalid handoff file digest entry")
        path = bundle / relative
        if (Path(relative).is_absolute() or ".." in Path(relative).parts or
                path.is_symlink() or not path.is_file() or
                not path.resolve().is_relative_to(bundle)):
            raise ValueError(f"handoff file is missing or unsafe: {relative}")
        if _sha(path) != digest:
            raise ValueError(f"handoff file hash changed: {relative}")
        total_bytes += path.stat().st_size
    if "archive_bytes" in manifest and (
            len(declared) != 1 or total_bytes != manifest["archive_bytes"]):
        raise ValueError("handoff archive byte count changed")
    return {"bundle": str(bundle), "schema": manifest.get("schema"),
            "files_verified": len(declared), "bytes_verified": total_bytes,
            "manifest_sha256": _sha(manifest_path),
            "scope": "manifest_declared_bytes_only_not_runtime_or_physical_validation"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.bundle_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
