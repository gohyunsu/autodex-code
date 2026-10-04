#!/usr/bin/env python3
"""Fail fast when the precision-insertion BODex/Franka environment is incomplete."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-no-gpu", action="store_true")
    args = parser.parse_args()

    import mujoco
    import cv2
    import msgpack
    import pymodbus
    import serial
    import torch
    import torch_scatter
    import zmq

    modules = [
        "coal_openmp_wrapper",
        "curobo.curobolib.geom_cu",
        "curobo.curobolib.kinematics_fused_cu",
        "curobo.curobolib.lbfgs_step_cu",
        "curobo.curobolib.line_search_cu",
        "curobo.curobolib.tensor_step_cu",
    ]
    for module in modules:
        importlib.import_module(module)

    cuda_available = torch.cuda.is_available()
    if not cuda_available and not args.allow_no_gpu:
        raise RuntimeError("PyTorch cannot access CUDA; run this check on the robot/GPU host")

    runtime_modules = ["chime"]
    skipped_runtime_modules = []
    if cuda_available:
        runtime_modules.append("src.execution.run_pipeline")
    else:
        # Importing cuRobo's MotionGen constructs CUDA tensors at module scope.
        # ``--allow-no-gpu`` is a CPU/package audit, so report this omission
        # explicitly instead of crashing before the flag can take effect.
        skipped_runtime_modules.append("src.execution.run_pipeline (requires CUDA)")
    for module in runtime_modules:
        importlib.import_module(module)

    gpu = None
    capability = None
    if cuda_available:
        gpu = torch.cuda.get_device_name(0)
        capability = list(torch.cuda.get_device_capability(0))
        value = torch.arange(16, device="cuda", dtype=torch.float32).sum().item()
        if value != 120.0:
            raise RuntimeError(f"unexpected CUDA tensor result: {value}")

    print(
        json.dumps(
            {
                "status": "PASS",
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": cuda_available,
                "gpu": gpu,
                "capability": capability,
                "torch_scatter": torch_scatter.__version__,
                "mujoco": mujoco.__version__,
                "opencv": cv2.__version__,
                "opencv_aruco": hasattr(cv2, "aruco"),
                "pyzmq": zmq.__version__,
                "msgpack": msgpack.__version__,
                "pymodbus": pymodbus.__version__,
                "pyserial": serial.__version__,
                "extensions": modules,
                "runtime_imports": runtime_modules,
                "skipped_runtime_imports": skipped_runtime_modules,
                "camera_sdk": {
                    "PySpin": bool(importlib.util.find_spec("PySpin")),
                    "required_on_robot_host_only_for_local_timestamp_monitor": True,
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
