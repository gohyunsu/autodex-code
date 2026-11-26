# MV-GoTrack and Qwen share capture4's GPU. cuRobo and Franka control stay on
# kds-Z790. The generated robot config points only at the experiment-owned
# /home/capture4/precision_insertion_qwen deployment; it never modifies a
# shared capture4 environment or camera service.
precision_experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$precision_experiment_dir/mvgotrack_axis_alignment_capture4.env.sh"
unset precision_experiment_dir

export PRECISION_CAPTURE4_QWEN_ROBOT_CONFIG=${PRECISION_CAPTURE4_QWEN_ROBOT_CONFIG:-$PRECISION_LOCAL_PROJECT_ROOT/runtime/franka_runtime_capture4_qwen.json}
export PRECISION_ROBOT_CONFIG=$PRECISION_CAPTURE4_QWEN_ROBOT_CONFIG
