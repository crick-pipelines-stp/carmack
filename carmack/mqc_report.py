"""Shared helpers for writing MultiQC custom-content JSON payloads.

Every reporting module under carmack builds its own MultiQC payloads inline --
this module supplies the two constants naming carmack's MultiQC parent section,
the writer that puts finished payloads on disk, and the helper that shapes a
linegraph's points, so each stage's reporting stays a sibling implementation
rather than a subclass.

``linegraph_xy_pairs`` earns its place here on the same grounds the constants
do, rather than because four callers happen to want the same few lines. The
shape a linegraph's ``data`` has to take is a fact about MultiQC's
custom-content input contract, and that contract is identical for every stage
carmack reports on; it is not a piece of any one stage's payload construction,
the way the counter being plotted and the titles above it are.

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


def linegraph_xy_pairs(counts: Mapping[int, int]) -> list[list[int]]:
    """Shape a numeric distribution into the ``[x, y]`` pairs MultiQC plots on a numeric axis.

    A linegraph's ``data`` can be handed over as a mapping of x to y, and that
    is the shape that looks natural in Python, but it is the shape that draws
    the wrong line. JSON object keys are strings, and MultiQC's custom-content
    linegraph path orders a mapping's keys lexically, so ``10`` sorts between
    ``1`` and ``2`` and a distribution reaching double digits zigzags. Sorting
    the mapping on carmack's side cannot help: the sort that decides the axis
    happens downstream of the handover, after carmack has stopped having a say.
    (MultiQC does compute a numeric coercion of the x values, but assigns it to
    a rebound attribute while passing the pre-coercion object on to the plot,
    so the line branch never sees it.)

    A list of ``[x, y]`` pairs takes the other branch. It is a first-class
    supported input shape that MultiQC recognises by type and builds the
    mapping from itself, which keeps the x values numeric -- and keeps them
    ``int`` rather than the floats the coercion would have produced.

    The inner pairs are ``list`` and not ``tuple`` on purpose: MultiQC's
    detection of the pair shape is ``isinstance(x_to_y[0], list)``, which a
    tuple fails in memory even though JSON would round-trip it as an array.

    The sorting is not what fixes the axis -- the shape is -- but it is done
    here anyway so the emitted JSON reads in the order the plot draws it and
    the payload on disk describes itself.

    An empty mapping raises rather than returning an empty list, because an
    empty pair list is the one input MultiQC cannot survive: it reaches for
    ``x_to_y[0]`` unguarded, raises ``IndexError``, and takes down the entire
    report rather than the one section. This is belt and braces -- every caller
    guards its counter and returns ``None`` before reaching here, so it is
    unreachable today -- and it exists so that a builder written later that
    forgets its guard fails loudly and attributably inside carmack instead of
    as a MultiQC crash, which is the same bargain ``write_mqc_payloads``
    already strikes.

    Args:
        counts: Observed y value per integer x value, in any order.

    Returns:
        One ``[x, y]`` list per entry, ordered ascending by x.

    Raises:
        ValueError: If ``counts`` is empty, which would render as a pair list
            MultiQC indexes without guarding.
    """
    if not counts:
        raise ValueError("MultiQC linegraph data cannot be built from an empty distribution")

    return [[x, counts[x]] for x in sorted(counts)]


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
    nothing matched, and MultiQC renders it. That check deliberately does not
    catch a per-sample empty list either -- a dict holding one is still truthy
    -- because the linegraph builders guard that case themselves by returning
    ``None``, so those guards are load-bearing and must not be deleted on the
    assumption that this writer would have caught it.

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
