"""Chemistry-and-arm-agnostic insert-boundary arithmetic shared by both trim arms.

``prepare-reads`` trims R1 down to its genomic/cDNA insert on two different arms — the
unmatched arm off the end of ``UMI_POS``, the matched arm off the end of ``TGIDX_POS`` —
and both need the same three-way answer to "where does the insert actually start". This
module holds that answer as one function, dispatching purely on the anchor component's
``type`` (absent, homopolymer, or fixed length) rather than any hardcoded component name
or chemistry constant, so neither arm needs its own copy of the arithmetic.
"""

from carmack.assign_targets.tgidx_locator import homopolymer_run_end
from carmack.chemistry.read_component import ReadComponent, ReadComponentType


def insert_start(reference: int, anchor: ReadComponent | None, seq: str) -> int:
    """Return the 0-based read coordinate where the insert begins.

    The caller already knows ``reference`` — the preceding component's span end — and
    hands over whichever right anchor its arm uses, resolved by ``umi_right_anchor()``
    or ``tgidx_right_anchor()``. From there the answer depends only on what kind of
    anchor sits between ``reference`` and the insert: no anchor means there is nothing
    to trim past, so ``reference`` already is the answer; a homopolymer anchor's real
    extent varies read to read with polymerase slippage, so it is read off ``seq`` by
    delegating to :func:`homopolymer_run_end`; every other anchor type has a length
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
        return homopolymer_run_end(seq, reference, anchor.homopolymer_base)
    return reference + anchor.length
