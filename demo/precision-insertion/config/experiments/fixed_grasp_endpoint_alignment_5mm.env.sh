# Controlled A/B baseline: fixed 5 mm fixture and grasp, endpoint/temporal
# pre-insertion alignment. Machine paths stay in zerodex_capture4.env.sh.
precision_experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$precision_experiment_dir/fixed_grasp_repeatability_5mm.env.sh"
unset precision_experiment_dir
export PRECISION_SAME_TRIAL_ALIGNMENT_SOURCE=endpoint
