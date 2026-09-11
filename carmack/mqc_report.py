"""Shared helper for writing MultiQC custom-content JSON payloads.

Every reporting module under carmack builds its own MultiQC payloads inline --
this module supplies only the two constants naming carmack's MultiQC parent
section and the thin JSON-writing helper, so each stage's reporting stays a
sibling implementation rather than a subclass.
"""

import json
from collections.abc import Mapping
from pathlib import Path

CARMACK_PARENT_ID = "carmack"
CARMACK_PARENT_NAME = "Carmack"


def write_mqc_json(path: str | Path, payload: Mapping[str, object]) -> None:
    """Write a MultiQC custom-content payload to disk as JSON.

    Args:
        path: Output file path. Its parent directory must already exist.
        payload: JSON-serialisable MultiQC custom-content payload.
    """
    with Path(path).open("w") as f:
        json.dump(payload, f, indent=2)
