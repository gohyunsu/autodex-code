#!/usr/bin/env python3
"""Extract one metric mesh from a sliced Bambu Studio 3MF archive.

The precision-key 0.1 mm source survived only inside a ``.gcode.3mf`` file.
This utility reads the embedded object XML, applies its component transform,
converts millimetres to metres, validates the mesh, and writes an STL.  It does
not reconstruct geometry from G-code and does not alter the model dimensions.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import trimesh


CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
PRODUCTION = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"


def _transform(text: str | None) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    if not text:
        return matrix
    values = np.asarray([float(value) for value in text.split()])
    if values.shape != (12,):
        raise ValueError(f"unsupported 3MF transform with {len(values)} values")
    # 3MF stores a 3x4 affine transform as row vectors.
    matrix[:3, :4] = values.reshape(4, 3).T
    return matrix


def extract(source: Path) -> trimesh.Trimesh:
    with zipfile.ZipFile(source) as archive:
        root = ET.fromstring(archive.read("3D/3dmodel.model"))
        component = root.find(
            f".//{{{CORE}}}component[@{{{PRODUCTION}}}path]"
        )
        if component is None:
            raise ValueError("3MF has no external object component")
        object_path = component.attrib[f"{{{PRODUCTION}}}path"].lstrip("/")
        component_transform = _transform(component.attrib.get("transform"))
        object_root = ET.fromstring(archive.read(object_path))

    vertices_node = object_root.find(f".//{{{CORE}}}vertices")
    triangles_node = object_root.find(f".//{{{CORE}}}triangles")
    if vertices_node is None or triangles_node is None:
        raise ValueError("3MF object does not contain one triangle mesh")
    vertices = np.asarray([
        [float(node.attrib[axis]) for axis in ("x", "y", "z")]
        for node in vertices_node
    ], dtype=np.float64)
    faces = np.asarray([
        [int(node.attrib[index]) for index in ("v1", "v2", "v3")]
        for node in triangles_node
    ], dtype=np.int64)
    homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
    vertices_mm = (component_transform @ homogeneous.T).T[:, :3]
    mesh = trimesh.Trimesh(vertices=vertices_mm * 1.0e-3, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight or mesh.volume <= 0.0:
        raise ValueError("extracted mesh is not a positive watertight solid")
    return mesh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        parser.error(f"source does not exist: {source}")
    mesh = extract(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output)
    print(output)
    print({
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "bounds_m": mesh.bounds.tolist(),
        "watertight": bool(mesh.is_watertight),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
