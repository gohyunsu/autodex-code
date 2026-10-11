"""Historical reset drift/tilt examples are opt-in diagnostics only."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnose_synthetic_repose import _optional_reset_fidelity  # noqa: E402


def test_synthetic_repose_defaults_to_unrestricted_stock_v8_reset_seeds():
    assert _optional_reset_fidelity(None, None) == (None, None)
    assert _optional_reset_fidelity(5.0, 10.0) == (.005, 10.0)


@pytest.mark.parametrize("drift,tilt", [
    (5.0, None), (None, 10.0), (0.0, 10.0), (5.0, -1.0),
])
def test_optional_reset_fidelity_requires_a_valid_pair(drift, tilt):
    with pytest.raises(ValueError):
        _optional_reset_fidelity(drift, tilt)
