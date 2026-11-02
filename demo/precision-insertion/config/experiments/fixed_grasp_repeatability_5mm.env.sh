# Controlled repeatability study: hold socket calibration and grasp identity
# fixed while re-observing the upright cylinder before every physical pickup.
# Machine paths and controller settings belong to zerodex_capture4.env.sh;
# this file contains only experiment variables.
export PRECISION_GAP_MM=${PRECISION_GAP_MM:-5}
export PRECISION_INITIAL_CANDIDATE=${PRECISION_INITIAL_CANDIDATE:-table/0/20195}
export PRECISION_FIXED_CANDIDATE=${PRECISION_FIXED_CANDIDATE:-1}
