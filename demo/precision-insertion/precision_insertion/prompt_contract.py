"""Reject stale color cues for the white precision key and socket."""

from __future__ import annotations

import re


_OTHER_COLOR = re.compile(
    r"\b(red|blue|green|yellow|orange|purple|pink|black|brown|gray|grey|"
    r"silver|gold|beige|cyan|magenta)\b",
    re.IGNORECASE,
)


def reject_incorrect_object_color(prompt: str) -> None:
    """Fail before capture if an object SAM prompt uses an incorrect color.

    The key and every socket in this task are white. A prompt can omit color,
    but an unrelated color word can redirect SAM to another board feature.
    """
    if not isinstance(prompt, str):
        raise TypeError("precision SAM prompt must be text")
    match = _OTHER_COLOR.search(prompt)
    if match is not None:
        raise ValueError(
            f"precision key and socket are white; remove {match.group()!r} "
            "from the SAM prompt")
