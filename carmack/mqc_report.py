"""Shared helpers for writing MultiQC custom-content JSON payloads.

Every reporting module under carmack builds its own MultiQC payloads inline --
this module supplies only the two constants naming carmack's MultiQC parent
section and the writer that puts finished payloads on disk, so each stage's
reporting stays a sibling implementation rather than a subclass.

``write_mqc_payloads`` is the writer a stage hands its finished payloads to.
It gives each payload a file of its own because MultiQC reads one
custom-content file as one section: it takes that file's top-level ``data``
and never walks nested payloads, so bundling two payloads under keys of a
stage's own choosing loses both charts, and loses them to a warning rather
than an error. Splitting the payloads inside the writer makes that rule
structural instead of something every call site has to remember.

The filename comes from the payload's own ``id`` for the same reason: the id
is what MultiQC anchors the section on, so deriving the name from it means
the file and the section it defines can never drift apart.
"""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

CARMACK_PARENT_ID = "carmack"
CARMACK_PARENT_NAME = "Carmack"


def write_mqc_payloads(
    output_dir: Path, prefix: str, payloads: Iterable[Mapping[str, object] | None]
) -> list[Path]:
    """Write each MultiQC payload to a file of its own, named after the payload's id.

    A payload of ``None`` is how a builder reports that it measured nothing,
    and is skipped: an empty chart claiming a distribution that was never
    observed is worse than an absent section. Any payload MultiQC would
    discard -- one that cannot name a file, or one carrying no top-level
    ``data`` -- raises instead of reaching disk, so the defect is loud and
    attributable rather than buried in a MultiQC warning.

    Only the top level of ``data`` is checked for emptiness. A payload mapping
    a sample to an empty mapping is a legitimate result for a run in which
    nothing matched, and MultiQC renders it.

    Every payload is checked before any of them is written, so a rejected
    payload leaves the output directory exactly as it found it. Validating as
    each file went out would instead leave the payloads ahead of the bad one on
    disk, which is the half-written report this writer exists to rule out.

    Args:
        output_dir: Directory the files are written into. It must already exist.
        prefix: Sample prefix leading every filename.
        payloads: MultiQC custom-content payloads, each optionally ``None``.

    Returns:
        The paths written, in payload order.

    Raises:
        ValueError: If a payload has no usable string ``id``, or no non-empty
            top-level ``data``.
    """
    # payloads may be a single-pass generator, so the validating pass keeps what it
    # accepts rather than leaving the writing pass with a consumed iterable.
    accepted: list[tuple[str, Mapping[str, object]]] = []
    for payload in payloads:
        if payload is None:
            continue

        payload_id = payload.get("id")
        if not isinstance(payload_id, str) or not payload_id:
            raise ValueError(f"MultiQC payload has no usable string id: {payload_id!r}")
        if not payload.get("data"):
            raise ValueError(f"MultiQC payload {payload_id} has no top-level data to render")

        accepted.append((payload_id, payload))

    written: list[Path] = []
    for payload_id, payload in accepted:
        stem = payload_id.removeprefix(f"{CARMACK_PARENT_ID}_")
        path = output_dir / f"{prefix}.{stem}_mqc.json"
        with path.open("w") as f:
            json.dump(payload, f, indent=2)
        written.append(path)

    return written
