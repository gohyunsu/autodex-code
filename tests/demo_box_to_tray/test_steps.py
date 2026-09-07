"""CLI wiring for the step sequence: what each step commands the runner to do."""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("trimesh")

from src.demo.box_to_tray.run_demo import (  # noqa: E402
    GOAL_TARGET,
    container_lift_height,
    filter_grasps,
    Step,
    default_prompt,
    parse_args,
    parse_fixture,
    parse_step,
    prompt_for,
)


def test_step_sequence_is_parsed_in_order():
    args = parse_args(["--fixture", "bowl=smallbowl",
                       "--step", "apple:bowl:tray", "--step", "banana:table:bowl"])
    assert args.steps == [Step("apple", "bowl", "tray"), Step("banana", "table", "bowl")]
    assert args.fixtures == {"bowl": "smallbowl"}


def test_single_object_shorthand_is_one_box_to_tray_step():
    args = parse_args(["--obj", "attached_container"])
    assert args.steps == [Step("attached_container", "box", "tray")]
    assert args.fixtures == {"box": "open_box"}


def test_shorthand_and_steps_cannot_be_mixed():
    with pytest.raises(SystemExit):
        parse_args(["--obj", "apple", "--step", "banana:table:bowl"])


def test_a_step_naming_an_unmeasured_container_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--step", "apple:bowl:tray"])          # no --fixture bowl=
    with pytest.raises(SystemExit):
        parse_args(["--step", "banana:table:bowl"])        # target not measured


def test_nothing_to_do_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--fixture", "bowl=smallbowl"])


def test_prompts_default_to_readable_names_and_can_be_overridden():
    args = parse_args(["--fixture", "bowl=smallbowl", "--step", "apple:bowl:tray",
                       "--prompt", "apple=red apple"])
    assert prompt_for(args, "apple") == "red apple"
    # SAM3 reads natural language, so an asset name loses its underscores.
    assert prompt_for(args, "smallbowl") == "smallbowl"
    assert default_prompt("open_box") == "open box"


def test_parse_helpers_reject_malformed_values():
    assert parse_fixture("bowl=smallbowl") == ("bowl", "smallbowl")
    assert parse_fixture("smallbowl") == ("smallbowl", "smallbowl")
    assert parse_step("apple:bowl:tray") == Step("apple", "bowl", "tray")
    for bad in ("apple:bowl", "apple::tray", "a:b:c:d"):
        with pytest.raises(Exception):
            parse_step(bad)


def test_goal_target_is_accepted_without_a_fixture():
    """`goal` is measured from the object itself in turn 1, not from a container."""
    args = parse_args(["--fixture", "bowl=smallbowl", "--step", "apple:bowl:goal"])
    assert args.steps == [Step("apple", "bowl", GOAL_TARGET)]
    assert "measured goal pose" in args.steps[0].describe()


def test_goal_defaults_place_precisely_and_container_drops_from_height():
    args = parse_args(["--fixture", "bowl=smallbowl",
                       "--step", "apple:bowl:goal", "--step", "banana:table:bowl"])
    # A goal step sets the object back down on its recorded contact height.
    assert args.goal_clearance < 0.01 and args.goal_pose_mode == "yaw"
    # A container step releases well above the rim so the object drops in.
    assert args.into_gap == pytest.approx(0.10)


class _Grasp:
    def __init__(self, source, episode):
        self.source, self.episode = source, episode


def test_exclude_grasp_drops_only_the_matching_library_entries():
    pool = [
        _Grasp("selected_100_inspire",
               "/home/robot/shared_data/autodex_dataset/selected_100_inspire/banana/20260406_001847"),
        _Grasp("selected_100_inspire",
               "/home/robot/shared_data/autodex_dataset/selected_100_inspire/banana/20260406_002001"),
        _Grasp("v8_inspire",
               "/home/robot/shared_data/AutoDex/experiment/v8/inspire/banana/20260812_123929"),
    ]
    kept, dropped = filter_grasps(pool, ["20260406_001847"])
    assert [g.episode for g in dropped] == [pool[0].episode]
    assert len(kept) == 2
    # A store-wide pattern works the same way.
    kept, dropped = filter_grasps(pool, ["selected_100"])
    assert len(kept) == 1 and kept[0].source == "v8_inspire"
    # No patterns is a no-op, not an empty pool.
    assert filter_grasps(pool, []) == (pool, [])


def test_container_lift_clears_the_rim_by_the_drop_gap():
    """The drop into a container happens from the carry height, so the lift
    has to reach it: a 15 cm lift off the table leaves a tall bowl's rim above
    the object, which turned the intended drop into a descent into the bowl."""
    args = parse_args(["--fixture", "bowl=smallbowl", "--step", "banana:table:bowl"])
    # Bowl rim measured at 0.133 m: 0.133 + 0.10 - 0.04 table = 0.193 m.
    assert container_lift_height(args, 0.133) == pytest.approx(0.193, abs=1e-9)
    # A shallow container never lowers the default lift.
    assert container_lift_height(args, 0.05) == pytest.approx(args.lift_height)
    # And an implausible rim cannot command an unbounded lift.
    assert container_lift_height(args, 1.5) == pytest.approx(args.max_lift_height)


def test_hold_retract_orientation_defaults_on():
    base = ["--fixture", "bowl=smallbowl", "--step", "banana:table:bowl"]
    assert parse_args(base).hold_retract_orientation is True
    assert parse_args(base + ["--no-hold-retract-orientation"]).hold_retract_orientation is False


def test_home_pose_choice():
    """clear_view is FR3_INIT with J0 -40 deg; init removes that swing."""
    base = ["--fixture", "bowl=smallbowl", "--step", "apple:bowl:goal"]
    assert parse_args(base).home_pose == "clear_view"
    assert parse_args(base + ["--home-pose", "init"]).home_pose == "init"


def test_home_before_step_can_be_turned_off():
    base = ["--fixture", "bowl=smallbowl", "--step", "apple:bowl:goal"]
    assert parse_args(base).home_before_step is True
    assert parse_args(base + ["--no-home-before-step"]).home_before_step is False


def test_a_failed_step_stops_the_round_by_default():
    """A skipped step looked like the robot ignoring it and taking the next object."""
    base = ["--fixture", "bowl=smallbowl", "--step", "apple:bowl:goal",
            "--step", "banana:table:bowl"]
    assert parse_args(base).stop_on_step_failure is True
    assert parse_args(base + ["--no-stop-on-step-failure"]).stop_on_step_failure is False


def test_auto_steps_is_opt_in_and_measures_containers_in_the_run_pass():
    """Objects already on the table means no per-step placement prompt."""
    base = ["--fixture", "bowl=smallbowl", "--step", "apple:bowl:tray"]
    assert parse_args(base).auto_steps is False
    auto = parse_args(base + ["--auto-steps"])
    assert auto.auto_steps is True
    # Containers are re-measured every round unless they are known not to move.
    assert auto.reuse_fixtures is False
    assert parse_args(base + ["--auto-steps", "--reuse-fixtures"]).reuse_fixtures is True


def test_tray_and_container_targets_select_different_release_modes():
    args = parse_args(["--fixture", "bowl=smallbowl",
                       "--step", "apple:bowl:tray", "--step", "banana:table:bowl"])
    # tray: a relative J0 turn, lay-down onto a stated surface height
    assert args.place_turn_deg == -30.0 and args.place_bearing_deg is None
    assert args.tray_top_z == pytest.approx(0.05)
    # container: released above the rim, which needs a positive gap
    assert args.into_gap > 0
