#!/usr/bin/env python3
"""Export or verify the exact cached Qwen snapshot used by the local VLM.

The Hugging Face cache snapshot contains symlinks into ``blobs/``.  This
handoff copies the dereferenced bytes into a standalone directory so another
PC can pass its ``payload/model`` path as ``vlm.model_id``.  It deliberately
does not package the separate ZeroDex source checkout or any credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
REQUIRED_FILES = frozenset({
    "config.json", "model.safetensors", "preprocessor_config.json",
    "tokenizer.json", "tokenizer_config.json",
})
SCHEMA = "precision_insertion_local_qwen_handoff_v1"


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _copy_and_hash(source: Path, target: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        for block in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
            outgoing.write(block)
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def export(snapshot_dir: Path, output_root: Path) -> dict:
    snapshot = Path(snapshot_dir).expanduser().resolve()
    target = Path(output_root).expanduser().resolve()
    working = target.with_name(target.name + ".incomplete")
    if target.exists() or working.exists():
        raise FileExistsError(f"handoff already exists: {target}")
    if not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    entries = sorted(snapshot.iterdir())
    if (not REQUIRED_FILES <= {path.name for path in entries} or
            any(not path.is_file() or path.name.startswith(".") for path in entries)):
        raise ValueError("Qwen snapshot has missing or unsupported root entries")
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl":
        raise ValueError("snapshot is not the configured Qwen3-VL model")
    model_dir = working / "payload" / "model"
    model_dir.mkdir(parents=True)
    files = {}
    for source in entries:
        size, digest = _copy_and_hash(source, model_dir / source.name)
        files[source.name] = {"bytes": size, "sha256": digest}
    manifest = {
        "schema": SCHEMA,
        "model_id": MODEL_ID,
        "source_revision": snapshot.name,
        "model_card": "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct",
        "model_card_license": "apache-2.0",
        "file_count": len(files),
        "files": files,
        "payload_relative_path": "payload/model",
        "scope": "standalone_model_files_not_vlm_accuracy_or_robot_readiness",
        "robot_ready": False,
    }
    with (working / "MANIFEST.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    working.rename(target)
    return manifest


def verify(output_root: Path) -> dict:
    target = Path(output_root).expanduser().resolve()
    manifest = json.loads((target / "MANIFEST.json").read_text(encoding="utf-8"))
    if (manifest.get("schema") != SCHEMA or
            manifest.get("model_id") != MODEL_ID or
            manifest.get("robot_ready") is not False or
            manifest.get("payload_relative_path") != "payload/model" or
            not isinstance(manifest.get("files"), dict) or
            not REQUIRED_FILES <= set(manifest["files"])):
        raise ValueError("invalid Qwen handoff manifest")
    model_dir = target / "payload" / "model"
    files = sorted(model_dir.iterdir())
    if ({path.name for path in files} != set(manifest["files"]) or
            any(not path.is_file() or path.is_symlink() for path in files) or
            manifest.get("file_count") != len(files)):
        raise ValueError("Qwen handoff file set differs from manifest")
    for path in files:
        size, digest = _hash_file(path)
        if manifest["files"][path.name] != {"bytes": size, "sha256": digest}:
            raise ValueError(f"Qwen handoff bytes differ: {path.name}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify-root", type=Path)
    group.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.verify_root is not None:
        if args.output_root is not None:
            parser.error("--output-root is only used when exporting")
        root = args.verify_root
        result = verify(root)
    else:
        if args.output_root is None:
            parser.error("export requires --output-root")
        root = args.output_root
        result = export(args.snapshot_dir, root)
    print(json.dumps({
        "root": str(Path(root).expanduser().resolve()),
        "model_id": result["model_id"],
        "source_revision": result["source_revision"],
        "file_count": result["file_count"],
        "total_bytes": sum(row["bytes"] for row in result["files"].values()),
        "robot_ready": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
