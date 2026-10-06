"""Batch-generate reorientation scenes for a given h value.

With ``--hand``, writes to the hand-specific runtime/BODex scene root
``AutoDex/scene/{hand}/{obj}/reorient_{cm}/{idx}.json``.  Without it, keeps the
legacy object-local output for backwards compatibility.
Each scene's meta records pose_i, pose_j, h, scene_idx for traceability.
"""
import argparse
import json
import sys
from pathlib import Path

from autodex.utils.path import get_obj_root, get_scene_dir

sys.path.insert(0, str(Path(__file__).parent))
from gen_scene import _obj_dir, gen_reorient_scene  # noqa: E402

OBJECTS = [
    "blue_alarm", "organizer_beige", "pepsi", "attached_container",
    "knife_sharpner", "pringles", "pepper_tuna", "soaptray",
    "potato_mesher", "bamboo_box",
    "icecream_scoop", "donut", "banana", "white_hand_shower",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", nargs="+", default=OBJECTS)
    parser.add_argument("--h", type=float, required=True,
                        help="h in meters (0.0, 0.04, 0.08)")
    parser.add_argument("--thickness", type=float, default=0.01)
    parser.add_argument("--version", default="v8",
                        help="v8 tabletop asset contract (only supported value)")
    parser.add_argument(
        "--hand", choices=["allegro", "inspire", "inspire_left"],
        help="write to the hand-specific runtime scene root",
    )
    parser.add_argument(
        "--target-obj",
        help="optional BODex contact proxy; valid only with one --objects entry",
    )
    args = parser.parse_args()
    if args.version != "v8":
        parser.error("gen_all supports only --version v8; legacy assets are not used")
    obj_root = get_obj_root(args.version)
    if args.target_obj and len(args.objects) != 1:
        parser.error("--target-obj requires exactly one --objects entry")

    h_cm = int(round(args.h * 100))
    scene_type = f"reorient_{h_cm}"

    grand_total = 0
    for obj in args.objects:
        tt_dir = _obj_dir(obj, obj_root) / "processed_data" / "info" / "tabletop"
        pose_ids = sorted([int(p.stem) for p in tt_dir.glob("*.npy")])
        output_obj = args.target_obj or obj
        out_dir = (
            Path(get_scene_dir(args.hand, output_obj, scene_type))
            if args.hand else _obj_dir(output_obj, obj_root) / "scene" / scene_type
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for i in pose_ids:
            for j in pose_ids:
                if i == j:
                    continue
                scene = gen_reorient_scene(obj, i, j, args.h,
                                            obj_root=obj_root,
                                            thickness=args.thickness,
                                            target_obj_name=args.target_obj)
                scene["meta"]["scene_type"] = scene_type
                with open(out_dir / f"{i}_{j}.json", "w") as f:
                    json.dump(scene, f, indent=2)
                count += 1
        grand_total += count
        print(f"{obj}: {count} scenes -> {out_dir}")
    print(f"\nDONE: {grand_total} scenes (scene_type={scene_type})")


if __name__ == "__main__":
    main()
