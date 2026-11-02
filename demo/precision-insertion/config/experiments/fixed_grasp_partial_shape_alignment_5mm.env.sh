# Controlled A/B variant: the same 5 mm fixture and grasp, preferring the
# multiview partial-cylinder pose and falling back to endpoint/temporal data.
precision_experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$precision_experiment_dir/fixed_grasp_repeatability_5mm.env.sh"
unset precision_experiment_dir
export PRECISION_SAME_TRIAL_ALIGNMENT_SOURCE=partial_shape
