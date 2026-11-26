# Active geometric calibration for the kds-Z790 + capture4 rig.
#
# Keep the selected dated solutions here, outside the control-flow code.  A
# run copies these exact paths into its launch snapshot, so later evidence can
# identify which calibration produced every projected point and robot pose.
precision_calibration_root="$HOME/calibration/zerodex_capture4"
export PRECISION_CALIB_DIR=${PRECISION_CALIB_DIR:-$precision_calibration_root/extrinsic/20261118_232125/solution_filtered}
export PRECISION_C2R_NPY=${PRECISION_C2R_NPY:-$precision_calibration_root/handeye/20261118_233621/solution/C2R.npy}
# Canonical trajectory for future hand-eye recapture.  This is independent of
# the commissioned runtime pair above.  The operator manually taught these 16
# stationary, non-duplicate poses after fixing the board in its current mount.
export PRECISION_SHARED_EXTRINSIC_CALIBRATION=${PRECISION_SHARED_EXTRINSIC_CALIBRATION:-$precision_calibration_root/extrinsic/20261118_232125}
export PRECISION_SHARED_HANDEYE_CALIBRATION=${PRECISION_SHARED_HANDEYE_CALIBRATION:-$precision_calibration_root/handeye/20261118_233621}
export PRECISION_HANDEYE_TRAJECTORY_SESSION=${PRECISION_HANDEYE_TRAJECTORY_SESSION:-$PRECISION_SHARED_HANDEYE_CALIBRATION/session}
unset precision_calibration_root
