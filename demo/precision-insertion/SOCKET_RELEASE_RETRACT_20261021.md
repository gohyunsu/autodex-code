# October 21 socket release retract diagnosis

The live run `20261021_104732_950124` stopped before the 20 mm axial insertion.
The episode had recorded `grasp_success=true` and `preinsert_reached=true`;
`insertion_success` was still unset. The operator saw the white key held clear
of the board and socket, with the board and socket unmoved. FR3 state feedback
after the exception showed a stationary preinsert arm.

`StockSocketRelease.prepare()` plans the hand opening, reverse socket-axis
exit, and clear-view retract before any insertion command. The saved
`plan_js_to_init` retract succeeded, but `socket_release._path()` raised
`clear-view retract lacks a dense open-hand FR3/Inspire path`. The stock
`GraspPlanner.plan_js_to_init` solves all 13 arm/hand joints together. It does
not constrain the six Inspire finger joints between its identical open-hand
endpoints. In three read-only replays with the exact saved socket/key scene,
cuRobo produced 123–159 waypoints and intermediate finger deviations of
0.105–0.124 rad. All endpoints were correct and the maximum joint step was
0.043 rad, below the configured 0.12 rad path limit. The failing condition
was the intermediate finger motion, not a missing arm path.

The release operation commands an open hand throughout the arm retract. The
planner result is now copied with its six hand joints fixed to `INSPIRE_INIT`
before the dense path check. The existing cuRobo sampled world/self-collision
check then validates that exact open-hand path against the socket and
hypothetically inserted key. In all three saved-scene replays, the frozen-hand
paths passed that collision check. If the frozen-hand path intersects the
scene, the normal collision check still rejects it.

This fix has not been physically tested. The current key remains in the hand
at preinsert until the operator resets the scene or an explicitly planned
continuation handles that state. Do not call a fresh session's `home()` while
the key remains held near the socket.

Regression test: `test_release_retract_holds_hand_open_despite_curobo_finger_wiggle`
in `tests/test_socket_release.py`; the focused socket-release suite passed
(5 tests), the complete precision-insertion suite passed (1067 passed,
4 skipped), and `run_robot_pc.sh --check` passed its static file checks. The
source change is confined to `precision_insertion/socket_release.py`.
