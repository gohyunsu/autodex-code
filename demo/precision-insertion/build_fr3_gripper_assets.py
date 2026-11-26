#!/usr/bin/env python3
"""Build the repository-owned FR3 + stock Franka gripper cuRobo assets.

The arm portion is copied from the already commissioned FR3/Inspire model;
the Inspire subtree is replaced by the standard Franka hand and two
independent prismatic finger joints.  The generated URDF intentionally keeps
both finger joints explicit because the runtime maps the daemon's total jaw
width to half-width per joint.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = (ROOT / "autodex/planner/src/curobo/content/assets/robot" /
              "fr3_gripper_description")
CONFIG_ROOT = ROOT / "autodex/planner/src/curobo/content/configs/robot"
SOURCE = (ROOT / "autodex/planner/src/curobo/content/assets/robot" /
          "fr3_inspire_description/fr3_inspire.urdf")
OUTPUT = ASSET_ROOT / "fr3_gripper.urdf"
FLOATING_OUTPUT = ASSET_ROOT / "franka_gripper_floating.urdf"
ROBOT_CONFIG_OUTPUT = CONFIG_ROOT / "fr3_gripper.yml"
HAND_CONFIG_OUTPUT = CONFIG_ROOT / "franka_gripper_floating.yml"
SPHERES_OUTPUT = CONFIG_ROOT / "spheres/fr3_gripper.yml"


def _origin(parent, *, xyz="0 0 0", rpy="0 0 0"):
    ET.SubElement(parent, "origin", {"xyz": xyz, "rpy": rpy})


def _mesh_link(robot, name: str, visual: str, *, collision: str):
    link = ET.SubElement(robot, "link", {"name": name})
    vis = ET.SubElement(link, "visual")
    _origin(vis)
    geom = ET.SubElement(vis, "geometry")
    ET.SubElement(geom, "mesh", {"filename": visual})
    col = ET.SubElement(link, "collision")
    _origin(col)
    geom = ET.SubElement(col, "geometry")
    ET.SubElement(geom, "mesh", {"filename": collision})
    return link


def _joint(robot, name: str, parent: str, child: str, *, joint_type: str,
           xyz: str, rpy: str = "0 0 0", axis: str | None = None):
    joint = ET.SubElement(robot, "joint", {"name": name, "type": joint_type})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    _origin(joint, xyz=xyz, rpy=rpy)
    if axis is not None:
        ET.SubElement(joint, "axis", {"xyz": axis})
        ET.SubElement(joint, "limit", {
            "effort": "20", "lower": "0", "upper": "0.04",
            "velocity": "0.2"})


def build() -> Path:
    source = ET.parse(SOURCE).getroot()
    robot = ET.Element("robot", {"name": "fr3_gripper"})
    # The mounted hand starts after the fixed fr3_link8 flange.  Preserve only
    # the arm subtree through that link/joint.
    allowed_links = {"base", *(f"fr3_link{i}" for i in range(9))}
    for node in source:
        if node.tag == "link" and node.attrib.get("name") in allowed_links:
            copied = deepcopy(node)
            for mesh in copied.findall(".//mesh"):
                filename = mesh.attrib.get("filename", "")
                if filename.startswith("./meshes/"):
                    mesh.attrib["filename"] = (
                        "../fr3_inspire_description/" + filename[2:])
            robot.append(copied)
        elif node.tag == "joint":
            child = node.find("child")
            if child is not None and child.attrib.get("link") in allowed_links:
                robot.append(deepcopy(node))

    mesh_root = ("../../../../../../../../src/validation/robothome/meshes/"
                 "robot_ee/franka_hand_white")
    _mesh_link(robot, "fr3_hand", f"{mesh_root}/visual/hand.dae",
               collision=f"{mesh_root}/collision/hand.stl")
    _joint(robot, "fr3_hand_joint", "fr3_link8", "fr3_hand",
           joint_type="fixed", xyz="0 0 0", rpy="0 0 -0.7853981633974483")
    _mesh_link(robot, "fr3_leftfinger", f"{mesh_root}/visual/finger.dae",
               collision=f"{mesh_root}/visual/finger.dae")
    _joint(robot, "fr3_finger_joint1", "fr3_hand", "fr3_leftfinger",
           joint_type="prismatic", xyz="0 0 0.0584", axis="0 1 0")
    _mesh_link(robot, "fr3_rightfinger", f"{mesh_root}/visual/finger.dae",
               collision=f"{mesh_root}/visual/finger.dae")
    _joint(robot, "fr3_finger_joint2", "fr3_hand", "fr3_rightfinger",
           joint_type="prismatic", xyz="0 0 0.0584",
           rpy="0 0 3.141592653589793", axis="0 1 0")

    ET.indent(robot, space="  ")
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(robot).write(OUTPUT, encoding="utf-8", xml_declaration=True)
    _build_floating_hand(robot)
    _write_configs()
    return OUTPUT


def _build_floating_hand(full_robot: ET.Element) -> None:
    robot = ET.Element("robot", {"name": "franka_gripper_floating"})
    ET.SubElement(robot, "link", {"name": "world"})
    parent = "world"
    root_joints = (
        ("x_joint", "x_link", "prismatic", "1 0 0"),
        ("y_joint", "y_link", "prismatic", "0 1 0"),
        ("z_joint", "z_link", "prismatic", "0 0 1"),
        ("x_rotation_joint", "rx_link", "revolute", "1 0 0"),
        ("y_rotation_joint", "ry_link", "revolute", "0 1 0"),
        ("z_rotation_joint", "fr3_hand_mount", "revolute", "0 0 1"),
    )
    for name, child, kind, axis in root_joints:
        ET.SubElement(robot, "link", {"name": child})
        joint = ET.SubElement(robot, "joint", {"name": name, "type": kind})
        ET.SubElement(joint, "parent", {"link": parent})
        ET.SubElement(joint, "child", {"link": child})
        _origin(joint)
        ET.SubElement(joint, "axis", {"xyz": axis})
        limit = ({"effort": "100", "lower": "-2", "upper": "2",
                  "velocity": "4"} if kind == "prismatic" else
                 {"effort": "100", "lower": "-6.28318530718",
                  "upper": "6.28318530718", "velocity": "4"})
        ET.SubElement(joint, "limit", limit)
        parent = child
    fixed = ET.SubElement(robot, "joint", {
        "name": "floating_to_hand", "type": "fixed"})
    ET.SubElement(fixed, "parent", {"link": parent})
    ET.SubElement(fixed, "child", {"link": "fr3_hand"})
    _origin(fixed)
    for name in ("fr3_hand", "fr3_leftfinger", "fr3_rightfinger"):
        robot.append(deepcopy(full_robot.find(f"link[@name='{name}']")))
    for name in ("fr3_finger_joint1", "fr3_finger_joint2"):
        robot.append(deepcopy(full_robot.find(f"joint[@name='{name}']")))
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(
        FLOATING_OUTPUT, encoding="utf-8", xml_declaration=True)


def _write_configs() -> None:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    SPHERES_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    arm_links = [f"fr3_link{i}" for i in range(8)]
    hand_links = ["fr3_hand", "fr3_leftfinger", "fr3_rightfinger"]
    links = arm_links + hand_links
    joints = [f"fr3_joint{i}" for i in range(1, 8)] + [
        "fr3_finger_joint1", "fr3_finger_joint2"]
    ignore = {
        "fr3_link0": ["fr3_link1"],
        "fr3_link1": ["fr3_link2"],
        "fr3_link2": ["fr3_link3"],
        "fr3_link3": ["fr3_link4"],
        "fr3_link4": ["fr3_link5"],
        "fr3_link5": ["fr3_link6", "fr3_link7"],
        "fr3_link6": ["fr3_link7", *hand_links],
        "fr3_link7": hand_links,
        "fr3_hand": ["fr3_leftfinger", "fr3_rightfinger"],
    }
    import yaml
    robot_cfg = {"robot_cfg": {"kinematics": {
        "use_usd_kinematics": False, "usd_path": "", "usd_robot_root": "",
        "isaac_usd_path": "", "usd_flip_joints": {},
        "usd_flip_joint_limits": [],
        "urdf_path": "robot/fr3_gripper_description/fr3_gripper.urdf",
        "asset_root_path": "robot/fr3_gripper_description",
        "base_link": "base", "ee_link": "fr3_hand",
        "link_names": links, "collision_link_names": links,
        "collision_spheres": "spheres/fr3_gripper.yml",
        "collision_sphere_buffer": 0.0,
        "self_collision_ignore": ignore,
        "self_collision_buffer": {name: 0.0 for name in links},
        "use_global_cumul": True, "mesh_link_names": links,
        "cspace": {
            "joint_names": joints,
            "retract_config": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571,
                               0.785, 0.04, 0.04],
            "cspace_distance_weight": [1.0] * 7 + [0.2, 0.2],
            "null_space_weight": [0.0] * 9,
            "max_jerk": 500.0, "max_acceleration": 15.0,
        },
    }}}
    hand_joint_names = ["x_joint", "y_joint", "z_joint",
                        "x_rotation_joint", "y_rotation_joint",
                        "z_rotation_joint", "fr3_finger_joint1",
                        "fr3_finger_joint2"]
    hand_cfg = {"robot_cfg": {"kinematics": {
        "use_usd_kinematics": False, "usd_path": "", "usd_robot_root": "",
        "isaac_usd_path": "", "usd_flip_joints": {},
        "usd_flip_joint_limits": [],
        "urdf_path": ("robot/fr3_gripper_description/"
                      "franka_gripper_floating.urdf"),
        "asset_root_path": "robot/fr3_gripper_description",
        "base_link": "world", "ee_link": "fr3_hand",
        "link_names": ["x_link", "y_link", "z_link", "rx_link", "ry_link",
                       "fr3_hand_mount", *hand_links],
        "collision_link_names": hand_links,
        "collision_spheres": "spheres/fr3_gripper.yml",
        "collision_sphere_buffer": 0.0,
        "self_collision_ignore": {
            "fr3_hand": ["fr3_leftfinger", "fr3_rightfinger"]},
        "self_collision_buffer": {name: 0.0 for name in hand_links},
        "use_global_cumul": True, "mesh_link_names": hand_links,
        "cspace": {
            "joint_names": hand_joint_names,
            "retract_config": [0.0] * 6 + [0.04, 0.04],
            "cspace_distance_weight": [1.0] * 6 + [0.2, 0.2],
            "null_space_weight": [0.0] * 8,
            "max_jerk": 500.0, "max_acceleration": 15.0,
        },
    }}}
    ROBOT_CONFIG_OUTPUT.write_text(
        yaml.safe_dump(robot_cfg, sort_keys=False), encoding="utf-8")
    HAND_CONFIG_OUTPUT.write_text(
        yaml.safe_dump(hand_cfg, sort_keys=False), encoding="utf-8")

    source_spheres = CONFIG_ROOT / "spheres/fr3_inspire.yml"
    prefix = source_spheres.read_text(encoding="utf-8").split(
        "  base_link:\n", 1)[0]
    # Reuse cuRobo's stock Franka hand profile with only link names changed.
    # The previous two large palm spheres extended well outside the actual
    # mesh and falsely rejected both supported tabletop grasps at the table.
    hand_spheres = """  fr3_hand:
  - center: [0.0, -0.075, 0.01]
    radius: 0.03
  - center: [0.0, -0.045, 0.01]
    radius: 0.03
  - center: [0.0, -0.015, 0.01]
    radius: 0.03
  - center: [0.0, 0.015, 0.01]
    radius: 0.03
  - center: [0.0, 0.045, 0.01]
    radius: 0.03
  - center: [0.0, 0.075, 0.01]
    radius: 0.03
  - center: [0.0, 0.065, -0.02]
    radius: 0.03
  - center: [0.0, -0.075, 0.03]
    radius: 0.03
  - center: [0.0, -0.045, 0.03]
    radius: 0.03
  - center: [0.0, -0.015, 0.03]
    radius: 0.03
  - center: [0.0, 0.015, 0.03]
    radius: 0.03
  - center: [0.0, 0.045, 0.03]
    radius: 0.03
  - center: [0.0, 0.075, 0.03]
    radius: 0.03
  - center: [0.0, -0.075, 0.05]
    radius: 0.02
  - center: [0.0, -0.045, 0.05]
    radius: 0.02
  - center: [0.0, -0.015, 0.05]
    radius: 0.02
  - center: [0.0, 0.015, 0.05]
    radius: 0.02
  - center: [0.0, 0.045, 0.05]
    radius: 0.02
  - center: [0.0, 0.075, 0.05]
    radius: 0.02
  fr3_leftfinger:
  - center: [0.0, 0.01, 0.04]
    radius: 0.01
  - center: [0.0, 0.01, 0.014]
    radius: 0.01
  fr3_rightfinger:
  - center: [0.0, -0.01, 0.04]
    radius: 0.01
  - center: [0.0, -0.01, 0.014]
    radius: 0.01
"""
    SPHERES_OUTPUT.write_text(prefix + hand_spheres, encoding="utf-8")


if __name__ == "__main__":
    print(build())
