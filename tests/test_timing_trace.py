import numpy as np

from autodex.executor.lift_policy import check_lift_start
from autodex.timing import TimingRecorder


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def test_trace_records_nested_strict_failure_snapshot():
    clock = FakeClock()
    trace = TimingRecorder(clock=clock)
    parent = trace.begin(phase="planning", kind="plan", name="lift_preflight")
    child = trace.begin(
        phase="planning", kind="plan", name="native_pose_constraint",
        parent_id=parent, constraint_mode="native_strict")
    clock.advance(0.25)
    trace.end(child, outcome="failure", failure_stage="motiongen_retry")
    clock.advance(0.05)
    trace.end(parent, outcome="failure")
    snapshot = trace.as_dict(trial_total_s=0.30)

    assert snapshot["schema_version"] == 1
    assert snapshot["spans"][1]["parent_id"] == snapshot["spans"][0]["id"]
    assert snapshot["spans"][1]["outcome"] == "failure"
    assert snapshot["spans"][1]["attributes"]["constraint_mode"] == "native_strict"
    assert snapshot["summary"]["unattributed_s"] == 0.0


def test_lift_gate_distinguishes_arm_and_hand_mismatch():
    expected = np.array([0.0, 0.1, -0.1, 0.2, 0.0, 0.1, 0.02, 0.03])
    hand_only = check_lift_start(
        expected[:6], expected, arm_dof=6,
        live_hand_qpos=np.array([0.22, 0.03]),
    )
    assert not hand_only.accepted
    assert hand_only.hand_checked
    assert not hand_only.hand_accepted
    assert hand_only.max_abs_rad == 0.0

    arm_only = check_lift_start(
        expected[:6] + np.array([0.2, 0, 0, 0, 0, 0]), expected,
        arm_dof=6, live_hand_qpos=expected[6:],
    )
    assert not arm_only.accepted
    assert arm_only.hand_accepted


def test_pipeline_outcome_snapshots_have_distinct_ownership():
    """The five critical routes must not collapse into one generic lift time."""
    clock = FakeClock()
    trace = TimingRecorder(clock=clock)
    cases = (
        ("strict_failure", "planning", "failure", "motiongen_retry"),
        ("candidate_reuse", "execution", "success", "candidate_preflight"),
        ("arm_mismatch", "execution", "success", "live_replan_arm_mismatch"),
        ("hand_mismatch", "execution", "success", "live_replan_hand_mismatch"),
        ("place_preflight_failure", "execution", "failure", "motiongen_first_attempt"),
    )
    for name, phase, outcome, source in cases:
        span = trace.begin(phase=phase, kind="plan", name=name)
        clock.advance(0.1)
        trace.end(span, outcome=outcome, lift_plan_source=source)
    snapshot = trace.as_dict(trial_total_s=0.5)
    observed = {
        item["name"]: (item["phase"], item["outcome"],
                       item["attributes"]["lift_plan_source"])
        for item in snapshot["spans"]
    }
    assert observed == {name: (phase, outcome, source)
                        for name, phase, outcome, source in cases}
