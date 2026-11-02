# Evidence-backed candidate policy for the ZeroDex 5 mm cylinder fixture.
# This is selection policy, not a generic grasp or motion rejection rule.
# Candidate 20096 intersected the socket at the 20 mm endpoint in two
# corrected-hold replays and left the released key outside in three trials.
export PRECISION_EXCLUDED_CANDIDATES=${PRECISION_EXCLUDED_CANDIDATES:-table/0/20096}
