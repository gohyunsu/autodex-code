#!/usr/bin/env python3
"""Create a self-contained precision-insertion handoff bundle.

The bundle preserves the runtime paths below ``shared_data`` while keeping
canonical source files, fabrication exports, reproducibility evidence, and
human-readable status separate.  The destination bundle must not already
exist; this exporter never merges into or replaces a previous handoff.

Typical NAS use on a host where ParaDex2 is mounted read-write::

    python scripts/precision_insertion/prepare_handoff.py \
      --output-root /mnt/paradex2/hyunsu

Use ``--minimal`` to omit the raw BODex/search evidence while retaining every
runtime asset needed by the current baseline.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


KEYS = (
    "precision_key_1p5mm",
    "precision_key_1p0mm",
    "precision_key_0p5mm",
    "precision_key_0p3mm",
)
PROXIES = (
    "precision_key_handle_contact_proxy",
    "precision_key_1p0mm_handle_contact_proxy",
    "precision_key_0p5mm_handle_contact_proxy",
    "precision_key_0p3mm_handle_contact_proxy",
)
SOCKET = "precision_socket_unified"


@dataclass(frozen=True)
class CopySpec:
    category: str
    source: Path
    destination: Path
    required: bool
    purpose: str


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            copy_function=shutil.copy2,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )
    else:
        shutil.copy2(source, destination)


def _files(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _specs(
    repo: Path,
    shared: Path,
    fabrication: Path,
    zerodex: Path,
    *,
    minimal: bool,
) -> list[CopySpec]:
    specs: list[CopySpec] = [
        CopySpec(
            "canonical_source",
            repo / "assets/precision_insertion",
            Path("source/autodex-code/assets/precision_insertion"),
            True,
            "Canonical STL inputs, camera profile, and BODex object lists.",
        ),
        CopySpec(
            "canonical_source",
            repo / "scripts/precision_insertion",
            Path("source/autodex-code/scripts/precision_insertion"),
            True,
            "Builders, validators, environment setup, and usage documentation.",
        ),
        CopySpec(
            "canonical_source",
            repo / "autodex/tasks",
            Path("source/autodex-code/autodex/tasks"),
            True,
            "Task outcome interface separating grasp and downstream task success.",
        ),
        CopySpec(
            "canonical_source",
            repo / "src/execution/run_auto.py",
            Path("source/autodex-code/src/execution/run_auto.py"),
            True,
            "Trial runner carrying session fixtures into every planning scene.",
        ),
        CopySpec(
            "canonical_source",
            repo / "src/execution/run_pipeline.py",
            Path("source/autodex-code/src/execution/run_pipeline.py"),
            True,
            "Integrated startup measurement and recovery pipeline.",
        ),
        CopySpec(
            "canonical_source",
            repo / "src/execution/scene_cfg.py",
            Path("source/autodex-code/src/execution/scene_cfg.py"),
            True,
            "Static socket collision-world injection.",
        ),
        CopySpec(
            "canonical_source",
            repo / "src/execution/session_fixtures.py",
            Path("source/autodex-code/src/execution/session_fixtures.py"),
            True,
            "SE(3) validation, medoid selection, and repeatability gate.",
        ),
        CopySpec(
            "canonical_source",
            repo / "docs/precision_insertion_pipeline_design.md",
            Path("source/autodex-code/docs/precision_insertion_pipeline_design.md"),
            True,
            "End-to-end experiment, controller, VLM, and failure-analysis design.",
        ),
        CopySpec(
            "canonical_source",
            repo / "docs/autodex_vs_precision_insertion.md",
            Path("source/autodex-code/docs/autodex_vs_precision_insertion.md"),
            True,
            "Evidence-scoped comparison of the original and precision-insertion pipelines.",
        ),
        CopySpec(
            "canonical_source",
            repo / "tests/test_precision_insertion_assets.py",
            Path("source/autodex-code/tests/test_precision_insertion_assets.py"),
            True,
            "Precision asset contract tests.",
        ),
        CopySpec(
            "canonical_source",
            repo / "tests/test_task_interface.py",
            Path("source/autodex-code/tests/test_task_interface.py"),
            True,
            "Task outcome interface tests.",
        ),
        CopySpec(
            "canonical_source",
            repo / "tests/test_session_fixtures.py",
            Path("source/autodex-code/tests/test_session_fixtures.py"),
            True,
            "Session fixture pose and collision-world contract tests.",
        ),
        CopySpec(
            "fabrication",
            fabrication,
            Path("fabrication/unified_single_socket"),
            True,
            "Printable STL and sliced 3MF files, including retained legacy variants.",
        ),
        CopySpec(
            "runtime",
            shared / "AutoDex/precision_insertion",
            Path("payload/shared_data/AutoDex/precision_insertion"),
            True,
            "Stage profiles, socket transforms, calibration snapshot, and visual evidence.",
        ),
        CopySpec(
            "runtime_reproducibility",
            shared / "AutoDex/contact_screen_staging/inspire/"
            "precision_insertion_tabletop_v1",
            Path("payload/shared_data/AutoDex/contact_screen_staging/inspire/"
                 "precision_insertion_tabletop_v1"),
            True,
            "Frame-aware tabletop contact candidates and corrected task-screen evidence.",
        ),
        CopySpec(
            "runtime_reproducibility",
            shared / "AutoDex/contact_screen_staging/inspire/"
            "precision_insertion_reset_12_handle_proxy_v3_frame_fixed",
            Path("payload/shared_data/AutoDex/contact_screen_staging/inspire/"
                 "precision_insertion_reset_12_handle_proxy_v3_frame_fixed"),
            True,
            "Two-contact AutoDex reorientation audit (37 numerical candidates, zero opposition passes).",
        ),
        CopySpec(
            "runtime_reproducibility",
            shared / "AutoDex/contact_screen_staging/inspire/"
            "precision_insertion_reset_12_handle_proxy_v4_three_finger_frame_fixed",
            Path("payload/shared_data/AutoDex/contact_screen_staging/inspire/"
                 "precision_insertion_reset_12_handle_proxy_v4_three_finger_frame_fixed"),
            True,
            "Three-contact force-cone follow-up (one numerical candidate, zero whole-hand/opposition passes).",
        ),
    ]

    for name in (*KEYS, *PROXIES, SOCKET):
        specs.append(
            CopySpec(
                "runtime",
                shared / "object_processing" / name,
                Path("payload/shared_data/object_processing") / name,
                True,
                "Metric object-processing asset.",
            )
        )
    for name in (*KEYS, *PROXIES):
        specs.append(
            CopySpec(
                "runtime",
                shared / "AutoDex/scene/inspire" / name,
                Path("payload/shared_data/AutoDex/scene/inspire") / name,
                True,
                "BODex scene definitions and controlled stable poses.",
            )
        )
    for name in KEYS:
        specs.append(
            CopySpec(
                "runtime",
                shared / "AutoDex/candidates/inspire/v8" / name,
                Path("payload/shared_data/AutoDex/candidates/inspire/v8") / name,
                True,
                "Historical candidate 78 pick/lift evidence; insertion-rejected by the new whole-hand gate.",
            )
        )
    specs.append(
        CopySpec(
            "runtime",
            shared / "AutoDex/bodex_raw/inspire/precision_insertion_v3_proxy/"
            "precision_key_handle_contact_proxy/table/0/84",
            Path("payload/shared_data/AutoDex/bodex_raw/inspire/"
                 "precision_insertion_v3_proxy/precision_key_handle_contact_proxy/"
                 "table/0/84"),
            True,
            "Source finger pose and contacts for the symmetry-derived rear insertion grasp preview.",
        )
    )
    for name in (*KEYS, SOCKET):
        specs.append(
            CopySpec(
                "runtime_gate",
                shared / "AutoDex/foundpose_assets" / name,
                Path("payload/shared_data/AutoDex/foundpose_assets") / name,
                True,
                "Expected FoundPose layout and generation-required marker.",
            )
        )

    if not minimal:
        specs.extend(
            [
                CopySpec(
                    "reproducibility",
                    shared / "AutoDex/bodex_raw/inspire/precision_insertion_v4_per_key_proxy",
                    Path("reproducibility/AutoDex/bodex_raw/inspire/precision_insertion_v4_per_key_proxy"),
                    False,
                    "Raw four-proxy BODex search used to obtain the historical pick/lift grasp.",
                ),
                CopySpec(
                    "reproducibility",
                    shared / "AutoDex/contact_screen_staging/inspire/precision_insertion_v4_common_grasp",
                    Path("reproducibility/AutoDex/contact_screen_staging/inspire/precision_insertion_v4_common_grasp"),
                    False,
                    "Declared-contact screening of historical grasp proposals.",
                ),
                CopySpec(
                    "reproducibility",
                    shared / "AutoDex/sim_filter_pass/inspire/precision_insertion_v4_common_grasp",
                    Path("reproducibility/AutoDex/sim_filter_pass/inspire/precision_insertion_v4_common_grasp"),
                    False,
                    "Historical MuJoCo/scene-clearance evidence; not a whole-hand insertion proof.",
                ),
            ]
        )

    zerodex_files = (
        "config/task_complete_checker/prompts.py",
        "config/task_complete_checker/result_parser.py",
        "run/components/task_complete_checker.py",
        "run/components/vlm_checker.py",
        "run/components/vlm_voting.py",
    )
    for relative in zerodex_files:
        specs.append(
            CopySpec(
                "zerodex_reference",
                zerodex / relative,
                Path("source/zerodex-reference") / relative,
                False,
                "Read-only reference for future VLM outcome evaluation.",
            )
        )
    return specs


def _readme(bundle_name: str, repo_info: dict, missing: list[dict]) -> str:
    missing_names = "\n".join(
        f"- `{item['source']}` — {item['purpose']}" for item in missing
    ) or "- none"
    return f"""# AutoDex precision-insertion handoff

Bundle: `{bundle_name}`

This directory is a transport bundle, not a writable experiment root. Verify
it first, then restore `payload/shared_data/` into the local writable
`~/shared_data` overlay. Keep raw experiment output local and publish a new
versioned handoff instead of editing this bundle in place.

## Code checkout

```text
repository: {repo_info['origin']}
branch: {repo_info['branch']}
commit: {repo_info['commit']}
```

```bash
git clone {repo_info['origin']} ~/autodex-code
cd ~/autodex-code
git checkout {repo_info['branch']}
```

## Verify and restore assets

```bash
cd /mnt/paradex2/hyunsu/{bundle_name}
sha256sum -c SHA256SUMS

# Inspect first; remove -n only after reviewing the destination.
rsync -ani --checksum payload/shared_data/ ~/shared_data/
rsync -ai  --checksum payload/shared_data/ ~/shared_data/
```

Then follow `source/autodex-code/scripts/precision_insertion/README.md` and run:

```bash
python scripts/precision_insertion/validate_assets.py
python scripts/precision_insertion/verify_autodex_camera_profile.py \
  --calib-dir <AUTODEX_CALIB_DIR> --require-runtime
```

## Directory map

- `source/`: canonical inputs/scripts and selected ZeroDex VLM reference code
- `fabrication/`: STL and sliced 3MF files; protocol uses 0.3/0.5/1.0/1.5 mm
- `payload/shared_data/`: files restored into the local AutoDex data overlay
- `reproducibility/`: raw BODex/search evidence; not needed for normal runtime
- `handoff_docs/OPEN_ITEMS.md`: work that cannot be represented as a file yet
- `MANIFEST.json`: source, purpose, required/optional status, sizes
- `SHA256SUMS`: byte-level integrity for every other file

## Source items unavailable while exporting

{missing_names}
"""


def _open_items() -> str:
    return """# Open items and ownership

## Blocking physical/runtime resources

1. Generate mesh-specific FoundPose `repre.pth` for all four keys and
   `precision_socket_unified` after restoring MV-GoTrack.
2. On the robot PC, audit the active AutoDex capture1/2/3/5/6 serial mapping,
   explicit intrinsics/extrinsics session, UTG900/timestamp configuration, and
   Franka hand-eye calibration.
3. At every `run_pipeline.py` session, measure the bolted fixture repeatedly
   with its generated socket representation and accept the generated
   `fixture_pose.session.json` only when its repeatability gate passes.
4. The corrected pose-004 audit has 10 whole-hand contact-policy passes and
   five sampled geometry passes through the 20 mm verification depth. These
   five still need continuous cuRobo, MuJoCo, and physical validation. All ten
   cross the table if the same grasp is held to the fully seated CAD pose, so
   full seating requires the separate release/retreat/guarded-press mode.
5. Generate at least one opposing-contact reorientation grasp and validate the
   complete Franka approach/lift/reorient/descent chain. The older key-only
   concepts are quarantined under the presentation audit folder.
6. Implement and commission the 1.5 mm insertion controller followed by the
   1.0 mm accuracy instrumentation and 0.5/0.3 mm force/contact search.
7. Add phase-aligned multi-view evidence and a read-only, shadow-mode VLM
   evaluator. Do not transplant ZeroDex's current task checker as a safety or
   success authority without a strict schema and an `unknown` result.

## Assets intentionally not fabricated

- FoundPose learned representations
- a reusable physical socket pose (intentionally forbidden; pose is per-session)
- physical grasp validation
- force limits, contact-search step sizes, or abort thresholds
- a robot-PC camera PASS audit fabricated from this development host

These omissions are recorded as gates so the pipeline fails closed.
"""


def build_bundle(
    *,
    output_root: Path,
    bundle_name: str,
    repo: Path,
    shared: Path,
    fabrication: Path,
    zerodex: Path,
    minimal: bool,
) -> Path:
    output_root = output_root.expanduser().resolve()
    final = output_root / bundle_name
    partial = output_root / f".{bundle_name}.partial-{os.getpid()}"
    if final.exists():
        raise FileExistsError(f"refusing to replace existing handoff: {final}")
    if partial.exists():
        raise FileExistsError(f"partial handoff already exists: {partial}")
    output_root.mkdir(parents=True, exist_ok=True)
    partial.mkdir()

    specs = _specs(repo, shared, fabrication, zerodex, minimal=minimal)
    records: list[dict] = []
    missing: list[dict] = []
    try:
        for spec in specs:
            source = spec.source.expanduser().resolve()
            record = {
                "category": spec.category,
                "source": str(source),
                "destination": str(spec.destination),
                "required": spec.required,
                "purpose": spec.purpose,
                "present": source.exists(),
            }
            if not source.exists():
                missing.append(record)
                records.append(record)
                if spec.required:
                    raise FileNotFoundError(f"required handoff source missing: {source}")
                continue
            _copy(source, partial / spec.destination)
            copied = partial / spec.destination
            record["bytes"] = sum(path.stat().st_size for path in _files(copied)) if copied.is_dir() else copied.stat().st_size
            record["files"] = len(list(_files(copied))) if copied.is_dir() else 1
            records.append(record)

        repo_info = {
            "origin": _git(repo, "remote", "get-url", "origin"),
            "branch": _git(repo, "branch", "--show-current"),
            "commit": _git(repo, "rev-parse", "HEAD"),
            "status_porcelain": _git(repo, "status", "--porcelain"),
        }
        zerodex_info = {
            "path": str(zerodex),
            "origin": _git(zerodex, "remote", "get-url", "origin") if (zerodex / ".git").is_dir() else None,
            "commit": _git(zerodex, "rev-parse", "HEAD") if (zerodex / ".git").is_dir() else None,
        }
        manifest = {
            "schema_version": 1,
            "bundle": bundle_name,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "repo": repo_info,
            "zerodex_reference": zerodex_info,
            "shared_root": str(shared.expanduser().resolve()),
            "minimal": minimal,
            "entries": records,
            "missing_optional_sources": missing,
            "known_runtime_blockers": [
                "FoundPose representations for four keys and socket",
                "continuous cuRobo and MuJoCo promotion of a 20 mm verification candidate",
                "commissioned release/retreat/guarded-press bundle for optional full seating",
                "opposing-contact robot reorientation candidate and full-chain plan",
                "accepted per-session socket pose measurement",
                "PASS AutoDex camera/calibration/hardware-sync/hand-eye audit",
                "commissioned insertion controllers and safety thresholds",
            ],
        }
        _write_json(partial / "MANIFEST.json", manifest)
        (partial / "README.md").write_text(
            _readme(bundle_name, repo_info, missing), encoding="utf-8"
        )
        docs = partial / "handoff_docs"
        docs.mkdir()
        (docs / "OPEN_ITEMS.md").write_text(_open_items(), encoding="utf-8")

        checksum_paths = [
            path for path in _files(partial)
            if path.name != "SHA256SUMS"
        ]
        lines = [
            f"{_sha256(path)}  {path.relative_to(partial).as_posix()}"
            for path in checksum_paths
        ]
        (partial / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
        partial.rename(final)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return final


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/paradex2/hyunsu"))
    parser.add_argument(
        "--bundle-name",
        default=f"autodex_precision_insertion_handoff_{dt.date.today().isoformat()}",
    )
    parser.add_argument("--repo-root", type=Path, default=repo)
    parser.add_argument("--shared-root", type=Path, default=Path.home() / "shared_data")
    parser.add_argument(
        "--fabrication-root",
        type=Path,
        default=Path.home() / "eraseme" / "unified_single_socket",
    )
    parser.add_argument(
        "--zerodex-root", type=Path, default=Path.home() / "realtime_vlm"
    )
    parser.add_argument(
        "--minimal", action="store_true",
        help="omit raw BODex/contact-screen/simulation search evidence",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_bundle(
        output_root=args.output_root,
        bundle_name=args.bundle_name,
        repo=args.repo_root.expanduser().resolve(),
        shared=args.shared_root.expanduser().resolve(),
        fabrication=args.fabrication_root.expanduser().resolve(),
        zerodex=args.zerodex_root.expanduser().resolve(),
        minimal=args.minimal,
    )
    print(bundle)


if __name__ == "__main__":
    main()
