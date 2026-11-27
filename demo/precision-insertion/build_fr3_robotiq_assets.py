#!/usr/bin/env python3
"""Build the commissioned FR3 + straight-mounted Robotiq 2F-85 assets.

The arm is reused from the repository FR3 model.  Robotiq geometry and mimic
kinematics come from the official robotiq_description package.  Only one
knuckle is independent, matching the physical rPR command/feedback channel.
"""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONTENT = ROOT / "autodex/planner/src/curobo/content"
ASSET_ROOT = CONTENT / "assets/robot/fr3_robotiq_2f85_description"
CONFIG_ROOT = CONTENT / "configs/robot"
SOURCE_ARM = CONTENT / "assets/robot/fr3_inspire_description/fr3_inspire.urdf"
VENDOR = Path(os.environ.get(
    "ROBOTIQ_DESCRIPTION_ROOT",
    "/home/hyunsu/cvpr27/rollout/vendor/robotiq-ros/grippers/"
    "robotiq_description"))
SOURCE_HAND = VENDOR / "urdf/robotiq_2f_85_viser.urdf"
OUTPUT = ASSET_ROOT / "fr3_robotiq_2f85.urdf"
FLOATING_OUTPUT = ASSET_ROOT / "robotiq_2f85_floating.urdf"
ROBOT_CONFIG_OUTPUT = CONFIG_ROOT / "fr3_robotiq_2f85.yml"
HAND_CONFIG_OUTPUT = CONFIG_ROOT / "robotiq_2f85_floating.yml"
SPHERES_OUTPUT = CONFIG_ROOT / "spheres/fr3_robotiq_2f85.yml"

HAND_LINKS = (
    "robotiq_85_base_link",
    "robotiq_85_left_knuckle_link", "robotiq_85_right_knuckle_link",
    "robotiq_85_left_finger_link", "robotiq_85_right_finger_link",
    "robotiq_85_left_inner_knuckle_link",
    "robotiq_85_right_inner_knuckle_link",
    "robotiq_85_left_finger_tip_link",
    "robotiq_85_right_finger_tip_link",
)
HAND_JOINTS = (
    "robotiq_85_left_knuckle_joint", "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_finger_joint", "robotiq_85_right_finger_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
)
MESH_STEMS = {
    "robotiq_85_base_link": "robotiq_base",
    "robotiq_85_left_knuckle_link": "left_knuckle",
    "robotiq_85_right_knuckle_link": "right_knuckle",
    "robotiq_85_left_finger_link": "left_finger",
    "robotiq_85_right_finger_link": "right_finger",
    "robotiq_85_left_inner_knuckle_link": "left_inner_knuckle",
    "robotiq_85_right_inner_knuckle_link": "right_inner_knuckle",
    "robotiq_85_left_finger_tip_link": "left_finger_tip",
    "robotiq_85_right_finger_tip_link": "right_finger_tip",
}


def _origin(parent, *, xyz="0 0 0", rpy="0 0 0"):
    ET.SubElement(parent, "origin", {"xyz": xyz, "rpy": rpy})


def _copy_arm(robot: ET.Element) -> None:
    source = ET.parse(SOURCE_ARM).getroot()
    allowed = {"base", *(f"fr3_link{i}" for i in range(9))}
    for node in source:
        if node.tag == "link" and node.attrib.get("name") in allowed:
            copied = deepcopy(node)
            for mesh in copied.findall(".//mesh"):
                filename = mesh.attrib.get("filename", "")
                if filename.startswith("./meshes/"):
                    mesh.attrib["filename"] = (
                        "../fr3_inspire_description/" + filename[2:])
            robot.append(copied)
        elif node.tag == "joint":
            child = node.find("child")
            if child is not None and child.attrib.get("link") in allowed:
                robot.append(deepcopy(node))


def _copy_meshes() -> None:
    for kind, suffix in (("visual", ".dae"), ("collision", ".stl")):
        destination = ASSET_ROOT / "meshes" / kind
        destination.mkdir(parents=True, exist_ok=True)
        for stem in MESH_STEMS.values():
            source = VENDOR / f"meshes/{kind}/2f_85/{stem}{suffix}"
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, destination / source.name)


def _copy_hand_nodes(robot: ET.Element) -> None:
    source = ET.parse(SOURCE_HAND).getroot()
    for name in HAND_LINKS:
        link = deepcopy(source.find(f"link[@name='{name}']"))
        if link is None:
            raise ValueError(f"official Robotiq URDF lacks {name}")
        for mesh in link.findall(".//mesh"):
            stem = MESH_STEMS[name]
            mesh.attrib["filename"] = f"./meshes/visual/{stem}.dae"
        collision = ET.SubElement(link, "collision")
        _origin(collision)
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "mesh", {
            "filename": f"./meshes/collision/{MESH_STEMS[name]}.stl"})
        robot.append(link)
    for name in HAND_JOINTS:
        joint = source.find(f"joint[@name='{name}']")
        if joint is None:
            raise ValueError(f"official Robotiq URDF lacks {name}")
        robot.append(deepcopy(joint))


def _mount(robot: ET.Element, parent: str) -> None:
    joint = ET.SubElement(robot, "joint", {
        "name": "fr3_robotiq_mount_joint", "type": "fixed"})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": "robotiq_85_base_link"})
    _origin(joint, rpy="0 0 1.5707963267948966")


def _build_full() -> ET.Element:
    robot = ET.Element("robot", {"name": "fr3_robotiq_2f85"})
    _copy_arm(robot)
    _copy_hand_nodes(robot)
    _mount(robot, "fr3_link8")
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(OUTPUT, encoding="utf-8", xml_declaration=True)
    return robot


def _build_floating(full: ET.Element) -> None:
    robot = ET.Element("robot", {"name": "robotiq_2f85_floating"})
    ET.SubElement(robot, "link", {"name": "world"})
    parent = "world"
    roots = (
        ("x_joint", "x_link", "prismatic", "1 0 0"),
        ("y_joint", "y_link", "prismatic", "0 1 0"),
        ("z_joint", "z_link", "prismatic", "0 0 1"),
        ("x_rotation_joint", "rx_link", "revolute", "1 0 0"),
        ("y_rotation_joint", "ry_link", "revolute", "0 1 0"),
        ("z_rotation_joint", "robotiq_mount", "revolute", "0 0 1"),
    )
    for name, child, kind, axis in roots:
        ET.SubElement(robot, "link", {"name": child})
        joint = ET.SubElement(robot, "joint", {"name": name, "type": kind})
        ET.SubElement(joint, "parent", {"link": parent})
        ET.SubElement(joint, "child", {"link": child})
        _origin(joint)
        ET.SubElement(joint, "axis", {"xyz": axis})
        ET.SubElement(joint, "limit", ({
            "effort": "100", "lower": "-2", "upper": "2", "velocity": "4"
        } if kind == "prismatic" else {
            "effort": "100", "lower": "-6.28318530718",
            "upper": "6.28318530718", "velocity": "4"}))
        parent = child
    fixed = ET.SubElement(robot, "joint", {
        "name": "floating_to_robotiq", "type": "fixed"})
    ET.SubElement(fixed, "parent", {"link": parent})
    ET.SubElement(fixed, "child", {"link": "robotiq_85_base_link"})
    _origin(fixed)
    for name in HAND_LINKS:
        robot.append(deepcopy(full.find(f"link[@name='{name}']")))
    for name in HAND_JOINTS:
        robot.append(deepcopy(full.find(f"joint[@name='{name}']")))
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(
        FLOATING_OUTPUT, encoding="utf-8", xml_declaration=True)


def _write_configs() -> None:
    arm_links = [f"fr3_link{i}" for i in range(8)]
    full_links = arm_links + list(HAND_LINKS)
    arm_joints = [f"fr3_joint{i}" for i in range(1, 8)]
    active = "robotiq_85_left_knuckle_joint"
    # The six mimic/fixed gripper links are parts of one known mechanism.  Do
    # not create artificial self-collision gates between those parts.
    ignore = {
        f"fr3_link{i}": ([f"fr3_link{i+1}"] if i < 7 else [])
        for i in range(8)
    }
    ignore["fr3_link5"] += ["fr3_link7", *HAND_LINKS]
    ignore["fr3_link6"] = ["fr3_link7", *HAND_LINKS]
    ignore["fr3_link7"] = list(HAND_LINKS)
    for name in HAND_LINKS:
        ignore[name] = [other for other in HAND_LINKS if other != name]
    common = {
        "use_usd_kinematics": False, "usd_path": "", "usd_robot_root": "",
        "isaac_usd_path": "", "usd_flip_joints": {},
        "usd_flip_joint_limits": [],
        "asset_root_path": "robot/fr3_robotiq_2f85_description",
        "ee_link": "robotiq_85_base_link",
        "collision_spheres": "spheres/fr3_robotiq_2f85.yml",
        "collision_sphere_buffer": 0.0,
        "use_global_cumul": True,
    }
    robot_cfg = {"robot_cfg": {"kinematics": {
        **common,
        "urdf_path": "robot/fr3_robotiq_2f85_description/fr3_robotiq_2f85.urdf",
        "base_link": "base", "link_names": full_links,
        "collision_link_names": full_links, "mesh_link_names": full_links,
        "self_collision_ignore": ignore,
        "self_collision_buffer": {name: 0.0 for name in full_links},
        "cspace": {
            "joint_names": arm_joints + [active],
            "retract_config": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571,
                               0.785, 0.0],
            "cspace_distance_weight": [1.0] * 7 + [0.2],
            "null_space_weight": [0.0] * 8,
            "max_jerk": 500.0, "max_acceleration": 15.0,
        },
    }}}
    floating_links = ["x_link", "y_link", "z_link", "rx_link", "ry_link",
                      "robotiq_mount", *HAND_LINKS]
    root_joints = ["x_joint", "y_joint", "z_joint", "x_rotation_joint",
                   "y_rotation_joint", "z_rotation_joint"]
    hand_cfg = {"robot_cfg": {"kinematics": {
        **common,
        "urdf_path": (
            "robot/fr3_robotiq_2f85_description/robotiq_2f85_floating.urdf"),
        "base_link": "world", "link_names": floating_links,
        "collision_link_names": list(HAND_LINKS),
        "mesh_link_names": list(HAND_LINKS),
        "self_collision_ignore": {
            name: [other for other in HAND_LINKS if other != name]
            for name in HAND_LINKS},
        "self_collision_buffer": {name: 0.0 for name in HAND_LINKS},
        "cspace": {
            "joint_names": root_joints + [active],
            "retract_config": [0.0] * 7,
            "cspace_distance_weight": [1.0] * 6 + [0.2],
            "null_space_weight": [0.0] * 7,
            "max_jerk": 500.0, "max_acceleration": 15.0,
        },
    }}}
    ROBOT_CONFIG_OUTPUT.write_text(
        yaml.safe_dump(robot_cfg, sort_keys=False), encoding="utf-8")
    HAND_CONFIG_OUTPUT.write_text(
        yaml.safe_dump(hand_cfg, sort_keys=False), encoding="utf-8")


def _write_spheres() -> None:
    """Voxelize actual link meshes into link-local collision spheres."""
    import trimesh

    source = CONFIG_ROOT / "spheres/fr3_inspire.yml"
    prefix = source.read_text(encoding="utf-8").split("  base_link:\n", 1)[0]
    sphere_map = {}
    for link, stem in MESH_STEMS.items():
        mesh = trimesh.load(
            ASSET_ROOT / f"meshes/collision/{stem}.stl", force="mesh")
        # A few hundred CAD-derived spheres preserve interactive cuRobo
        # latency; a 7 mm voxelization produced >1,700 spheres for this hand.
        pitch = 0.018 if link == "robotiq_85_base_link" else 0.014
        points = mesh.voxelized(pitch).fill().points
        # Half a voxel diagonal covers the voxel without inventing a large
        # palm envelope.  The resulting model follows the supplied CAD.
        radius = float(pitch * (3.0 ** 0.5) / 2.0)
        sphere_map[link] = [{
            "center": [round(float(value), 6) for value in point],
            "radius": round(radius, 6),
        } for point in points]
    hand_yaml = yaml.safe_dump({"collision_spheres": sphere_map},
                               sort_keys=False)
    body = hand_yaml.split("collision_spheres:\n", 1)[1]
    SPHERES_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SPHERES_OUTPUT.write_text(prefix + body, encoding="utf-8")


def build() -> Path:
    if not SOURCE_HAND.is_file():
        raise FileNotFoundError(
            f"Robotiq description not found; set ROBOTIQ_DESCRIPTION_ROOT: {VENDOR}")
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    _copy_meshes()
    full = _build_full()
    _build_floating(full)
    _write_configs()
    _write_spheres()
    return OUTPUT


if __name__ == "__main__":
    print(build())
