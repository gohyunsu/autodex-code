# ZeroDex capture4/kds-Z790 machine profile. Source from a shell before
# starting the namespaced capture4 manager, or let run_zerodex_capture4.sh
# source it. Each setting can be overridden by the caller's environment.
# Resolve relative to this file so the profile follows a separate checkout.
precision_profile_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
precision_profile_root=$(cd -- "$precision_profile_dir/../../.." && pwd)

export PRECISION_REPO_ROOT=${PRECISION_REPO_ROOT:-$precision_profile_root}
export PRECISION_STAGE_POLICY=${PRECISION_STAGE_POLICY:-$precision_profile_root/demo/precision-insertion/config/zerodex_capture4_stage_policy.json}
export PRECISION_LOCAL_PROJECT_ROOT=${PRECISION_LOCAL_PROJECT_ROOT:-$HOME/precision_insertion_project}
# This rig's NAS namespace is fixed rather than caller-overridable. Local
# restored copies are still allowed because they do not resolve below this
# NAS mount.
export PRECISION_NAS_MOUNT_ROOT=/mnt/paradex2
export PRECISION_ALLOWED_NAS_ROOT=/mnt/paradex2/hyunsu
export PRECISION_LEGACY_ASSET_ROOTS=${PRECISION_LEGACY_ASSET_ROOTS:-$HOME/shared_data/AutoDex:/home/robot/shared_data/AutoDex:$HOME/NAS-ParaDex2/AutoDex}
export PRECISION_NAS_PROJECT_ROOT=${PRECISION_NAS_PROJECT_ROOT:-/mnt/paradex2/hyunsu/precision_insertion}
export PRECISION_SHARED_ROOT=${PRECISION_SHARED_ROOT:-$PRECISION_NAS_PROJECT_ROOT/runtime/active_shared_data}
# The AutoDex directory name inside this project is the v8 asset schema.
# This profile never reads or writes the old ~/shared_data/AutoDex tree.
export PRECISION_STAGE_ROOT=${PRECISION_STAGE_ROOT:-$PRECISION_SHARED_ROOT/AutoDex/precision_insertion/zerodex_capture4}
export PRECISION_ROBOT_CONFIG=${PRECISION_ROBOT_CONFIG:-$PRECISION_STAGE_ROOT/franka_runtime.json}
export PRECISION_RUN_OUTPUT_ROOT=${PRECISION_RUN_OUTPUT_ROOT:-$PRECISION_NAS_PROJECT_ROOT/runs/zerodex_capture4}
export PRECISION_RECORDING_RECEIPT_ROOT=${PRECISION_RECORDING_RECEIPT_ROOT:-$PRECISION_RUN_OUTPUT_ROOT/_recording_receipts}
export PRECISION_PARADEX_ROOT=${PRECISION_PARADEX_ROOT:-$HOME/precision_insertion_project/vendor/paradex}
export PRECISION_DEV_ROOT=${PRECISION_DEV_ROOT:-$PRECISION_NAS_PROJECT_ROOT/development/zerodex_capture4}
export PRECISION_TIMING_LOCAL_ROOT=${PRECISION_TIMING_LOCAL_ROOT:-/tmp/precision-insertion-timing}
# Continue the human-reviewed episode collection on this rig. VLM answers are
# still stored beside manual labels; callers can explicitly choose vlm_only.
export PRECISION_LABEL_MODE=${PRECISION_LABEL_MODE:-manual}
# ZeroDex runs reobserve and correct XY at pre-insertion by default. An
# explicitly empty value disables this mode for a baseline run.
export PRECISION_SAME_TRIAL_XY_TOLERANCE_MM=${PRECISION_SAME_TRIAL_XY_TOLERANCE_MM-2.5}
# Take a new image after each held-key correction; one correction may move up
# to 10 mm, while the loop has no cumulative XY cap.
export PRECISION_SAME_TRIAL_MAX_STEP_MM=${PRECISION_SAME_TRIAL_MAX_STEP_MM-10}
# If the corrected 20 mm endpoint/path does not plan, retry the same measured
# hold at this shallower insertion depth.  Keep it explicit in the run profile
# rather than burying the experiment setting in Python control code.
export PRECISION_SAME_TRIAL_AXIAL_FALLBACK_DEPTH_MM=${PRECISION_SAME_TRIAL_AXIAL_FALLBACK_DEPTH_MM-10}
export PRECISION_SAME_TRIAL_ALIGNMENT_SOURCE=${PRECISION_SAME_TRIAL_ALIGNMENT_SOURCE-endpoint}
# Align the symmetry-reduced cylinder axis in the same photograph/move/
# rephotograph loop. These values are convergence and motion-step settings,
# not extra grasp/insertion rejection criteria.
export PRECISION_SAME_TRIAL_AXIS_TOLERANCE_DEG=${PRECISION_SAME_TRIAL_AXIS_TOLERANCE_DEG-2}
export PRECISION_SAME_TRIAL_MAX_TILT_STEP_DEG=${PRECISION_SAME_TRIAL_MAX_TILT_STEP_DEG-3}
# A straight raise still leaves the downward-facing key tip hidden in all
# four calibrated views of the saved 5/10 mm holds. Keep this extra motion
# opt-in until a raised live snapshot demonstrates a usable metric point.
export PRECISION_SAME_TRIAL_OBSERVATION_RAISE_MM=${PRECISION_SAME_TRIAL_OBSERVATION_RAISE_MM-0}
# An optional cylinder observation turn can be tried after no metric point is
# found. Sixty degrees planned in saved 5 and 10 mm episodes; live visibility
# is unknown, so keep it off until a photographed A/B comparison is intended.
export PRECISION_SAME_TRIAL_OBSERVATION_YAW_DEG=${PRECISION_SAME_TRIAL_OBSERVATION_YAW_DEG-0}
# If the key end is not measurable above the socket, make one short active
# observation move in the socket plane.  The direction is ranked from all
# calibrated cameras for key/rim pixel separation; the key returns before
# insertion.  This is an observation action, not a random insertion search.
export PRECISION_SAME_TRIAL_OBSERVATION_LATERAL_MM=${PRECISION_SAME_TRIAL_OBSERVATION_LATERAL_MM-10}
export PRECISION_SAME_TRIAL_OBSERVATION_LATERAL_ATTEMPTS=${PRECISION_SAME_TRIAL_OBSERVATION_LATERAL_ATTEMPTS-1}
export PRECISION_EXTRA_PYTHONPATH=${PRECISION_EXTRA_PYTHONPATH:-$precision_profile_root/demo/precision-insertion/compat}
export PRECISION_PLANNER_PY=${PRECISION_PLANNER_PY:-$HOME/.venvs/precision-vlm/bin/python}
export PRECISION_CHECK_PY=${PRECISION_CHECK_PY:-$PRECISION_PLANNER_PY}
export PRECISION_SAM_PYTHON=${PRECISION_SAM_PYTHON:-$PRECISION_PLANNER_PY}
precision_local_sam="$HOME/precision_insertion_project/cache/models/sam3.pt"
precision_default_sam="$PRECISION_NAS_PROJECT_ROOT/assets/shared_data/AutoDex/weights/sam3/sam3.pt"
if [[ -f $precision_local_sam ]]; then
  precision_default_sam=$precision_local_sam
fi
export PRECISION_SAM_CHECKPOINT=${PRECISION_SAM_CHECKPOINT:-$precision_default_sam}
precision_local_qwen="$PRECISION_LOCAL_PROJECT_ROOT/cache/models/Qwen3-VL-2B-Instruct"
if [[ -d $precision_local_qwen ]]; then
  export PRECISION_VLM_MODEL_ID=${PRECISION_VLM_MODEL_ID:-$precision_local_qwen}
fi
export PRECISION_FR3_INSPIRE_URDF=${PRECISION_FR3_INSPIRE_URDF:-$precision_profile_root/autodex/planner/src/curobo/content/assets/robot/fr3_inspire_description/fr3_inspire.urdf}
# Resolve the planner's cuRobo copy before the BODex copy on this profile.
export PRECISION_CUROBO_PYTHONPATH=${PRECISION_CUROBO_PYTHONPATH:-$precision_profile_root/autodex/planner/src}

export PRECISION_CAPTURE4_SSH_HOST=${PRECISION_CAPTURE4_SSH_HOST:-capture4@147.46.219.136}
export PRECISION_CAPTURE4_SSH_PORT=${PRECISION_CAPTURE4_SSH_PORT:-77}
export PRECISION_CAPTURE4_IP=${PRECISION_CAPTURE4_IP:-147.46.219.136}
export PRECISION_CAPTURE4_HOME=${PRECISION_CAPTURE4_HOME:-/home/capture4}
export PRECISION_CAPTURE4_ADAPTER_DIR=${PRECISION_CAPTURE4_ADAPTER_DIR:-$PRECISION_CAPTURE4_HOME/precision_insertion_capture4_20261027}
export PRECISION_CAPTURE4_ASSET_ROOT=${PRECISION_CAPTURE4_ASSET_ROOT:-$PRECISION_CAPTURE4_ADAPTER_DIR/cache/shared_data}
export PRECISION_CAPTURE4_PROJECT_ROOT=${PRECISION_CAPTURE4_PROJECT_ROOT:-$PRECISION_CAPTURE4_HOME/shared_data/hyunsu/precision_insertion}
export PRECISION_ZERODEX_FR3_FCI_IP=${PRECISION_ZERODEX_FR3_FCI_IP:-172.16.1.11}
export PRECISION_ZERODEX_DAEMON_HOST=${PRECISION_ZERODEX_DAEMON_HOST:-127.0.0.1}
export PRECISION_CAPTURE4_CAMERA_IDS=${PRECISION_CAPTURE4_CAMERA_IDS:-26053248,25322639,25305462,25322642}
# Feed all commissioned cameras into ChArUco triangulation. That estimator
# already refits each corner after reprojection outlier removal; the later
# socket-only fit independently records its per-capture centre inliers. Do not
# encode a permanent camera blacklist in the machine profile.
export PRECISION_BOARD_FIT_CAMERA_IDS=${PRECISION_BOARD_FIT_CAMERA_IDS:-$PRECISION_CAPTURE4_CAMERA_IDS}
export PRECISION_VLM_CAMERA_IDS=${PRECISION_VLM_CAMERA_IDS:-26053248,25322639,25305462,25322642}
# By default each session's socket-pose inliers are ordered by their measured
# centre residual. Set an explicit comma-separated order only for a controlled
# camera-priority experiment; no serial is permanently preferred.
export PRECISION_XY_PREFERRED_CAMERA_IDS=${PRECISION_XY_PREFERRED_CAMERA_IDS-}
# Use the three lowest-residual inliers from this session's socket-only fit.
# The capture and stored evidence still contain all four views. An experiment
# may explicitly set PRECISION_FINAL_VLM_CAMERA_IDS to replay a fixed subset.
export PRECISION_FINAL_VLM_VIEW_COUNT=${PRECISION_FINAL_VLM_VIEW_COUNT:-3}
export PRECISION_VISIBLE_GRIP_CAMERA_IDS=${PRECISION_VISIBLE_GRIP_CAMERA_IDS:-$PRECISION_VLM_CAMERA_IDS}

# Existing local CUDA headers are needed by nvdiffrast's first-use build.
export CUDA_HOME=${CUDA_HOME:-$HOME/miniconda3/envs/autodex_bodex}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6}
export MAX_JOBS=${MAX_JOBS:-4}
precision_cuda_headers="$CUDA_HOME/lib/python3.10/site-packages/nvidia"
for precision_header_dir in cuda_runtime cusparse cublas cusolver; do
  precision_include="$precision_cuda_headers/$precision_header_dir/include"
  if [[ -d "$precision_include" ]]; then
    export CPATH="$precision_include${CPATH:+:$CPATH}"
  fi
done
unset precision_profile_dir precision_profile_root precision_cuda_headers precision_header_dir precision_include precision_local_sam precision_default_sam precision_local_qwen
