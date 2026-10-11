#!/usr/bin/env python3
"""Archive freshly generated canonical FoundPose objects to a new NAS handoff.

The destination must not exist.  This command never promotes a representation
as metrically accurate; it only copies the generated files and checks the
large representation payloads against their local SHA256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from generate_missing_foundpose import MISSING


def _sha(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def publish(*, shared_root: Path, output_root: Path) -> dict:
    """Write a non-overwriting tar plus a read-back-verified manifest."""
    shared = Path(shared_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    sources = []
    for name in MISSING:
        directory = shared / "AutoDex/foundpose_assets" / name
        provenance = directory / "ONBOARDING_PROVENANCE.json"
        representation = (directory / "object_repre/v1" / name /
                          "1/repre.pth")
        record = json.loads(provenance.read_text(encoding="utf-8"))
        if (record.get("object") != name or
                record.get("physical_pose_qualified") is not False or
                not representation.is_file()):
            raise ValueError(f"incomplete or unqualified source: {name}")
        with representation.open("rb") as stream:
            digest = _sha(stream)
        if digest != record.get("representation_sha256"):
            raise ValueError(f"local representation differs from provenance: {name}")
        sources.append((name, directory, representation, record))

    output.mkdir(parents=True, exist_ok=False)
    archive = output / "foundpose_generated_7objects.tar"
    with tarfile.open(archive, mode="w") as handle:
        for name, directory, _, _ in sources:
            handle.add(directory, arcname=f"AutoDex/foundpose_assets/{name}")

    rows = []
    with tarfile.open(archive, mode="r") as handle:
        for name, _, representation, record in sources:
            member = handle.getmember(
                f"AutoDex/foundpose_assets/{name}/object_repre/v1/"
                f"{name}/1/repre.pth")
            stream = handle.extractfile(member)
            if stream is None:
                raise ValueError(f"NAS tar lacks representation: {name}")
            with stream:
                archive_digest = _sha(stream)
            if archive_digest != record["representation_sha256"]:
                raise ValueError(f"NAS tar representation hash mismatch: {name}")
            rows.append({
                "object": name,
                "source": str(representation),
                "representation_sha256": archive_digest,
                "physical_pose_qualified": False,
            })
    with archive.open("rb") as stream:
        archive_digest = _sha(stream)
    manifest = {
        "schema": "precision_insertion_generated_foundpose_handoff_v1",
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256_read_back": archive_digest,
        "objects": rows,
        "status": "generated_and_reloaded_not_metric_pose_validated",
        "physical_pose_qualified": False,
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = publish(**vars(args))
    print(json.dumps({
        "archive": str(Path(args.output_root) / report["archive"]),
        "archive_bytes": report["archive_bytes"],
        "archive_sha256_read_back": report["archive_sha256_read_back"],
        "objects": len(report["objects"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
