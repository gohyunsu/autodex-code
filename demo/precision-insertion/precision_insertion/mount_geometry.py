"""FR3 flange-to-Inspire transform from the selected URDF fixed joints.

The precision runner selects its URDF with PRECISION_FR3_INSPIRE_URDF. Parsing
the same file for the executor keeps measured-wrist feedback and cuRobo FK in
one frame. No camera, daemon, or robot is accessed here.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def link7_to_hand(urdf_path: Path) -> np.ndarray:
    joints = {
        joint.attrib["name"]: joint
        for joint in ET.parse(urdf_path).getroot().findall("joint")
    }

    def fixed_pose(name: str, parent: str, child: str) -> np.ndarray:
        joint = joints[name]
        if (joint.attrib["type"] != "fixed" or
                joint.find("parent").attrib["link"] != parent or
                joint.find("child").attrib["link"] != child):
            raise ValueError(f"{name} does not connect {parent} to {child}")
        origin = joint.find("origin")
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(origin.attrib["rpy"], sep=" ")).as_matrix()
        pose[:3, 3] = np.fromstring(origin.attrib["xyz"], sep=" ")
        return pose

    return (fixed_pose("fr3_joint8", "fr3_link7", "fr3_link8") @
            fixed_pose("flange_to_hand", "fr3_link8", "base_link"))
