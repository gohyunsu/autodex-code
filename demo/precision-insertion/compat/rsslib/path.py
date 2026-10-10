"""Fail-closed fallback for the unused legacy BODex output-root import.

Always pass ``generate.py --output_dir``. If its loader unexpectedly falls
back to ``bodex_path``, this invalid location fails rather than silently
writing into another user's legacy output tree.
"""

bodex_path = "/dev/null/precision_insertion_requires_explicit_output_dir"

__all__ = ["bodex_path"]
