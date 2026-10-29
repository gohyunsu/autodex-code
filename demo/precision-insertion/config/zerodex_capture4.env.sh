# ZeroDex capture4/kds-Z790 machine profile. Source from a shell before
# starting the namespaced capture4 manager, or let run_zerodex_capture4.sh
# source it. Each setting can be overridden by the caller's environment.
# Resolve relative to this file so the profile follows a separate checkout.
precision_profile_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
precision_profile_root=$(cd -- "$precision_profile_dir/../../.." && pwd)

export PRECISION_REPO_ROOT=${PRECISION_REPO_ROOT:-$precision_profile_root}
export PRECISION_STAGE_ROOT=${PRECISION_STAGE_ROOT:-$HOME/shared_data/AutoDex/precision_insertion/zerodex_capture4}
export PRECISION_PARADEX_ROOT=${PRECISION_PARADEX_ROOT:-$HOME/precision_insertion_project/vendor/paradex}
export PRECISION_DEV_ROOT=${PRECISION_DEV_ROOT:-$HOME/precision_insertion_project/development/zerodex_capture4}
export PRECISION_TIMING_LOCAL_ROOT=${PRECISION_TIMING_LOCAL_ROOT:-$PRECISION_DEV_ROOT/timing}
export PRECISION_NAS_PROJECT_ROOT=${PRECISION_NAS_PROJECT_ROOT:-/mnt/paradex2/hyunsu/precision_insertion}
# Continue the human-reviewed episode collection on this rig. VLM answers are
# still stored beside manual labels; callers can explicitly choose vlm_only.
export PRECISION_LABEL_MODE=${PRECISION_LABEL_MODE:-manual}
# ZeroDex runs reobserve and correct XY at pre-insertion by default. An
# explicitly empty value disables this mode for a baseline run.
export PRECISION_SAME_TRIAL_XY_TOLERANCE_MM=${PRECISION_SAME_TRIAL_XY_TOLERANCE_MM-2.5}
export PRECISION_SAME_TRIAL_OBSERVATION_RAISE_MM=${PRECISION_SAME_TRIAL_OBSERVATION_RAISE_MM-40}
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
export PRECISION_BOARD_FIT_CAMERA_IDS=${PRECISION_BOARD_FIT_CAMERA_IDS:-26053248,25322639,25322642}
export PRECISION_VLM_CAMERA_IDS=${PRECISION_VLM_CAMERA_IDS:-26053248,25322639,25305462,25322642}
# Four saved same-trial snapshots agreed on these three key-face views.
# The fourth view is processed only when the first three do not yield metric XY.
export PRECISION_XY_PREFERRED_CAMERA_IDS=${PRECISION_XY_PREFERRED_CAMERA_IDS:-25305462,25322639,25322642}
export PRECISION_PREINSERT_AUDIT_OVERLAY=${PRECISION_PREINSERT_AUDIT_OVERLAY:-0}
# Six human-labeled 5 mm socket episodes replayed correctly with a short A/B
# question and these three views. XY grounding still has all four available.
export PRECISION_FINAL_VLM_CAMERA_IDS=${PRECISION_FINAL_VLM_CAMERA_IDS:-26053248,25322639,25305462}
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
unset precision_profile_dir precision_profile_root precision_cuda_headers precision_header_dir precision_include precision_local_sam precision_default_sam
