# Active geometric calibration for the kds-Z790 + capture4 rig.
#
# Keep the selected dated solutions here, outside the control-flow code.  A
# run copies these exact paths into its launch snapshot, so later evidence can
# identify which calibration produced every projected point and robot pose.
precision_calibration_root="$PRECISION_NAS_PROJECT_ROOT/calibration/zerodex_capture4"
export PRECISION_CALIB_DIR=${PRECISION_CALIB_DIR:-$precision_calibration_root/extrinsic_solutions/20261103_board11_4cam_v1}
export PRECISION_C2R_NPY=${PRECISION_C2R_NPY:-$precision_calibration_root/handeye_solutions/20261103_board2_ee_fr3_replay_v1/C2R.npy}
unset precision_calibration_root
