"""Task-agnostic research tools for zero-shot precision execution."""

from .program import (
    ProgramValidationError,
    are_programs_isomorphic,
    validate_program,
    validate_zero_shot_contract,
)

__all__ = [
    "ProgramValidationError",
    "are_programs_isomorphic",
    "validate_program",
    "validate_zero_shot_contract",
]
