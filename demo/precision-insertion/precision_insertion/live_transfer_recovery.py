"""Close an uncertain transfer with supervised recovery and fresh key capture.

No key pose is inferred from a failed transfer. The injected recovery producer
must place the key on the board, fully open the hand, clear the camera view and
write the existing supervised-reset record. This fallback does not masquerade
as a planned automatic robot return.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping
from uuid import uuid4

from .action_loop import ActionHandler
from .live_key_trial import capture_returned_key_for_reset
from .session_policy import SessionDecision


@dataclass
class LiveTransferRecoveryHandlers:
    live: object
    recover_to_tabletop: Callable
    state_for_observation: Callable
    key_prompt: str
    capture_parameters: Mapping[str, object]
    landing_parameters: Mapping[str, object]
    capture_id_factory: Callable[[], str] = lambda: f"transfer_reset_{uuid4().hex}"

    def recover(self, runner, decision: SessionDecision) -> None:
        if (runner is not self.live.opened.runner or
                not isinstance(decision, SessionDecision) or
                runner.current_decision() != decision or
                decision.action != "recover_key_then_reobserve"):
            raise ValueError("transfer recovery belongs to another decision")
        recovery_log = Path(self.recover_to_tabletop(
            runner, decision)).expanduser().resolve()
        if not recovery_log.is_file():
            raise FileNotFoundError("transfer recovery returned no completion log")
        self.live.clear_view.before_key_capture()
        launch = self.live.cameras.launch
        capture_id = self.capture_id_factory()
        landed = capture_returned_key_for_reset(
            runner=runner, clear_view=self.live.clear_view,
            init_orchestrator=self.live.cameras.init_orchestrator,
            acquisition_metadata_for_request=(
                self.live.cameras.acquisition_metadata_for_request),
            state_for_observation=self.state_for_observation,
            capture_root=launch.capture_root,
            key_evidence_dir=(runner.output_dir / "transfer_reset_key_captures" /
                              capture_id),
            capture_id=capture_id,
            calibrated_camera_ids=set(launch.camera_ids),
            intrinsics_full=launch.intrinsics_full,
            extrinsics_full=launch.extrinsics_full,
            image_hw=launch.image_hw, key_prompt=self.key_prompt,
            view_limits=launch.view_limits,
            expected_action="recover_key_then_reobserve",
            **dict(self.capture_parameters))
        runner.observe_reset_landing(
            key_observation=landed.observation,
            key_evidence_dir=landed.key_evidence_dir,
            recovery_log_path=recovery_log,
            timestamp_s=landed.observation.acquisition_interval_s[1],
            **dict(self.landing_parameters))

    def handlers(self) -> dict[str, ActionHandler]:
        return {"recover_key_then_reobserve": self.recover}
