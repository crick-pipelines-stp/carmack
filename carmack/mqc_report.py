"""Shared identifiers and JSON writer for MultiQC custom content reports.

Every carmack stage that emits a MultiQC custom content section attaches
itself to one shared parent module, keyed by ``CARMACK_PARENT_ID`` and
``CARMACK_PARENT_NAME``, so MultiQC groups every stage's sections under one
Carmack heading rather than scattering them across the report.
``write_mqc_json`` is the one place a caller writes an MQC-readable JSON
payload to disk, so every stage serializes its report the same way.
"""

import json
from pathlib import Path
from typing import Any

CARMACK_PARENT_ID = "carmack"
CARMACK_PARENT_NAME = "Carmack"


def write_mqc_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a MultiQC custom content payload to disk as JSON.

    Args:
        path: Location to write the JSON file to.
        payload: The MultiQC custom content payload to serialize.
    """
    with open(path, "w") as handle:
        json.dump(payload, handle)
