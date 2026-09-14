"""Read-only candidate-selection policies for the Jacobian lift experiment.

The production v8 runner mixes immutable candidate geometry with mutable
``result.json`` records.  That is appropriate for collection, but reusing a
legacy success record while evaluating a replacement lift backend would hide
the very candidates the new backend might recover.  This module makes the
choice explicit without writing to either production or candidate state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional


CandidateKey = tuple[str, str, str]


@dataclass(frozen=True)
class CandidatePolicy:
    """Candidate-state semantics used for one offline trial.

    ``clean-state`` reproduces the v8 *catalogue and coverage ordering* as if
    no grasp has succeeded yet.  It deliberately ignores legacy result files.
    ``current-state`` reads the same shared completion state as production;
    it answers the narrower question of what a new planner would see today.
    ``verified-only`` retains only candidate directories whose existing
    ``result.json`` says ``success: true``.  It keeps the immutable coverage
    ranking, rather than treating those successes as completed coverage.
    ``pipeline-parity`` has the same mutable candidate-state read semantics
    as ``current-state``.  Its additional contract is enforced by
    ``run_session``: live FoundPose input, the production scene builder, and
    production early scene-skip/reorient decisions must be used.  No policy
    writes a result file into the candidate tree.
    """

    name: str
    skip_done: bool
    skip_scenes_with_success: bool
    success_only: bool
    success_root: Optional[str]
    mutable_state_source: str

    def as_dict(self) -> dict:
        return asdict(self)


def make_candidate_policy(name: str, *, clean_state_root: Path) -> CandidatePolicy:
    if name == "clean-state":
        # This unique, intentionally empty path makes ``load_coverage_map``
        # calculate its score from immutable ``covers`` lists only.  It is not
        # created and no candidate result is ever written there.
        return CandidatePolicy(
            name=name,
            skip_done=False,
            skip_scenes_with_success=False,
            success_only=False,
            success_root=str(clean_state_root),
            mutable_state_source="empty_experiment_state",
        )
    if name == "current-state":
        return CandidatePolicy(
            name=name,
            skip_done=True,
            skip_scenes_with_success=True,
            success_only=False,
            success_root=None,
            mutable_state_source="shared_candidate_result_json",
        )
    if name == "verified-only":
        # ``load_candidate(success_only=True)`` is the membership gate.  Its
        # coverage ranking must still come from immutable coverage data: using
        # the shared state here would assign every verified grasp zero remaining
        # coverage and silently empty the pool before that gate runs.
        return CandidatePolicy(
            name=name,
            skip_done=False,
            skip_scenes_with_success=False,
            success_only=True,
            success_root=str(clean_state_root),
            mutable_state_source="candidate_result_json_success_true",
        )
    if name == "pipeline-parity":
        # Candidate membership deliberately matches an ordinary, non-isolated
        # ``run_auto`` trial.  This is intentionally *not* success-only: both
        # prior successes and prior failures are completed records that the
        # production collection loop skips.  The live-input/scene half of the
        # contract is enforced at the run-session entry point, not here.
        return CandidatePolicy(
            name=name,
            skip_done=True,
            skip_scenes_with_success=True,
            success_only=False,
            success_root=None,
            mutable_state_source="shared_candidate_result_json_production_state",
        )
    raise ValueError(f"unknown candidate policy: {name}")


def order_coverage_keys(coverage_map: Mapping[CandidateKey, int]) -> list[CandidateKey]:
    """Mirror ``run_auto``'s descending remaining-coverage ordering.

    Python sorting is stable, so ties retain the JSON/map insertion order just
    as the production runner does.  Keys with zero remaining coverage are not
    candidate options in coverage mode.
    """

    useful = {tuple(map(str, key)): int(value)
              for key, value in coverage_map.items() if int(value) > 0}
    return sorted(useful, key=lambda key: -useful[key])


def coverage_metadata(coverage_map: Mapping[CandidateKey, int],
                      order: list[CandidateKey]) -> dict:
    normalized = {tuple(map(str, key)): int(value)
                  for key, value in coverage_map.items()}
    return {
        "n_coverage_candidates": len(normalized),
        "n_coverage_useful": sum(value > 0 for value in normalized.values()),
        "n_coverage_zero": sum(value == 0 for value in normalized.values()),
        "ordered_keys": [list(key) for key in order],
        "remaining_coverage_by_key": {
            "/".join(key): value for key, value in normalized.items()
        },
    }
