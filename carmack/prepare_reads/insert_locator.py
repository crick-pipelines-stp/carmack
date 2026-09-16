"""Chemistry-and-arm-agnostic insert-boundary arithmetic shared by both trim arms.

``prepare-reads`` trims R1 down to its genomic/cDNA insert on two different arms — the
unmatched arm off the end of ``UMI_POS``, the matched arm off the end of ``TGIDX_POS`` —
and both need the same three-way answer to "where does the insert actually start". This
module holds that answer as one function, dispatching purely on the anchor component's
``type`` (absent, homopolymer, or fixed length) rather than any hardcoded component name
or chemistry constant, so neither arm needs its own copy of the arithmetic.

Both arms then need a second answer laid over the first: whether the read has an insert
at all. The coordinate :func:`insert_start` returns is free to reach or pass the end of
the read, and a read cut there yields an empty sequence rather than a short one, so
:func:`insert_not_sequenced` sits beside it as the guard both arms put over the cut
before anything is written. The module therefore answers two questions for both arms —
where the insert starts, and whether there is one at all — and answers both from the cut
and the read alone.
"""

from carmack.assign_targets.tgidx_locator import locate_anchor_run
from carmack.chemistry.read_component import ReadComponent, ReadComponentType


def insert_start(reference: int, anchor: ReadComponent | None, seq: str) -> int:
    """Return the 0-based read coordinate where the insert begins.

    The caller already knows ``reference`` — the preceding component's span end — and
    hands over whichever right anchor its arm uses, resolved by ``umi_right_anchor()``
    or ``tgidx_right_anchor()``. From there the answer depends only on what kind of
    anchor sits between ``reference`` and the insert: no anchor means there is nothing
    to trim past, so ``reference`` already is the answer; a homopolymer anchor's real
    extent varies read to read with polymerase slippage, so it is read off ``seq`` by
    delegating to :func:`locate_anchor_run`; every other anchor type has a length
    fixed by the chemistry, so the answer is plain addition with no need, or reason, to
    consult ``seq`` at all.

    Args:
        reference: A read coordinate the caller already knows — the anchor's preceding
            component's span end.
        anchor: The right-anchor component for this arm, or ``None`` when the read
            structure names no anchor at all.
        seq: The full read sequence.

    Returns:
        The 0-based index in ``seq`` where the insert starts.
    """
    if anchor is None:
        return reference
    if anchor.type is ReadComponentType.HOMOPOLYMER:
        return locate_anchor_run(seq, reference, anchor.homopolymer_base, anchor.min_run).end
    return reference + anchor.length


def insert_not_sequenced(cut: int, seq: str) -> bool:
    """Return whether the read has no insert left once it is cut at ``cut``.

    On a 2-colour instrument the absence of signal is read as a base call rather than as
    nothing, so a cluster that dies just after the scaffold comes back with the rest of
    the read as a tail of the anchor base. The scan behind :func:`insert_start` then
    walks that run to the read's last base and reports the read end as where the insert
    begins, which is the honest answer: nothing past the scaffold was measured. Cutting
    there yields an empty sequence, and an empty FASTQ record is syntactically valid —
    its sequence and quality lines agree at length zero — so it clears every framing and
    length check downstream and desyncs the next reader that meets it. The caller asks
    this before writing anything, and drops the read instead.

    The gate belongs here and not inside the scan. Returning the read end for a run that
    saturates is :func:`locate_anchor_run`'s intended and tested contract, and its other
    reuse site — ``assign-targets``, which cuts a match window rather than an insert —
    guards that value in its own terms at its own call site. This module is the second
    reuse site, so it needs its own guard; making the scan clamp or refuse instead would
    take a decision that depends on what the coordinate is being used for away from the
    two callers that know.

    ``>=`` rather than ``==`` because only one of the two branches saturates. The
    homopolymer branch reads the run off ``seq`` and so can stop at exactly ``len(seq)``,
    but the fixed-length branch is arithmetic on a chemistry-declared length that never
    consults ``seq`` at all, and on a read that ended early it returns a coordinate
    strictly past the end. ``>=`` is therefore precisely the condition "``seq[cut:]`` is
    the empty string", which is the thing the caller is about to write.

    Quality is neither consulted nor taken as a parameter: FASTQ guarantees
    ``len(qual) == len(seq)``, so a cut that empties one empties the other, and the
    record is dropped whole rather than in parts.

    Args:
        cut: The 0-based read coordinate the insert would start at, as returned by
            :func:`insert_start`. Allowed to reach or exceed ``len(seq)``.
        seq: The full read sequence, untrimmed.

    Returns:
        ``True`` when nothing was sequenced after the cut, so the read has no insert to
        write; ``False`` when any insert remains, however short.
    """
    return cut >= len(seq)
