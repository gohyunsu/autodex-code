"""Offline, interactive validation of a Jacobian-continuation lift.

This package is deliberately separate from the execution pipeline.  It may
reuse perception and planner *inputs*, but never imports or constructs an arm
executor.
"""
