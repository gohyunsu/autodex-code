"""The stock capture command channel prints failures instead of raising."""

import logging

import numpy as np

from autodex.perception.init_orchestrator import InitOrchestrator


def test_live_init_reports_capture_command_failure(caplog):
    class Sender:
        def send_command(self, command, *, wait, cmd_info):
            assert command == "init" and wait is False
            print("[capture3] 'init' failed after 2 tries: Resource temporarily unavailable")
            print("[capture1] Command 'init' executed")

    orchestrator = object.__new__(InitOrchestrator)
    orchestrator.pc_list = ["capture1", "capture3"]
    orchestrator.cmd = Sender()
    with caplog.at_level(logging.WARNING):
        orchestrator.init_object(
            obj_name="key", mesh_path="/tmp/key.obj", assets_root="/tmp/assets",
            intrinsics_full={"camera": {
                "K_orig": np.eye(3), "K_undist": np.eye(3),
                "dist_params": np.zeros(5), "width": 4, "height": 4}},
            extrinsics_full={"camera": np.eye(4)}, image_hw=(4, 4),
            mode="live", load_silhouette=False)
    assert "[capture3] 'init' failed after 2 tries" in caplog.text
    assert "[capture1] Command 'init' executed" not in caplog.text
