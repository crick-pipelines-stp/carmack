"""The per-read scTIP/bowtie2 write path: header rendering plus paired bucket writes.

A matched read's target index selects one already-open (R1, R2) stream pair out of a
bucket map keyed by target index, and both streams need the same read name written
onto them. This module renders that name and performs the paired write, deliberately
using a pipe-delimited ``read_id|CB=<barcode>|UR=<umi>`` convention rather than
``ReadAnnotation.render()``'s space-separated one: SAM/BAM QNAME truncates at the first
whitespace, and this header must survive alignment intact for downstream scTIP tooling
to recover the cell barcode and UMI from it.

Neither function here re-derives what a target index bucket or a valid barcode set is:
header rendering delegates entirely to ``ChemistryBase.construct_full_barcode`` for the
barcode half, and the writer only ever looks a target index up in a bucket map that its
caller already built.
"""

from dataclasses import dataclass
from typing import Protocol

from carmack.chemistry.chemistry_base import ChemistryBase
from carmack.io.fastq_file import FastqFile


class WritableByteStream(Protocol):
    """Structural type for an already-open, write-only binary stream.

    Matches both a production ``carmack.io.subprocess_stream.SubprocessStream`` (from
    ``GzipFile.open_write_stream()``) and a plain ``io.BytesIO`` used in tests -- the
    only capability this module needs from either is ``write(bytes)``.
    """

    def write(self, data: bytes) -> object:
        """Write ``data`` to the stream."""


@dataclass(frozen=True)
class SctipBucketWriters:
    """Holds one scTIP target index bucket's already-open (R1, R2) write streams."""

    r1: WritableByteStream
    r2: WritableByteStream


def render_sctip_header(
    chemistry: ChemistryBase, read_id: str, barcodes: dict[str, str], umi: str
) -> str:
    """Render the pipe-delimited read name scTIP downstream tooling expects.

    The barcode half is exactly ``chemistry.construct_full_barcode(barcodes)`` -- this
    function does not reimplement barcode concatenation or its ordering, and does not
    catch the ``ValueError`` that call raises for a missing required barcode component.
    The UMI half is the ``umi`` argument passed through completely unmodified: no case
    normalisation, padding, or trimming.

    Args:
        chemistry: The chemistry whose read structure determines barcode component
            order.
        read_id: The read's identifier, without any leading ``@`` or trailing
            whitespace.
        barcodes: Barcode component name to sequence value, as required by
            ``construct_full_barcode``.
        umi: The UMI sequence, used verbatim.

    Returns:
        ``"{read_id}|CB=<full barcode>|UR=<umi>"`` with no embedded whitespace.

    Raises:
        ValueError: Propagated unmodified from ``construct_full_barcode`` when
            ``barcodes`` is missing a required component.
    """
    full_barcode = chemistry.construct_full_barcode(barcodes)
    return f"{read_id}|CB={full_barcode}|UR={umi}"


def write_sctip_read(
    writers: dict[str, SctipBucketWriters],
    tgidx: str,
    header: str,
    r1_seq: str,
    r1_qual: str,
    r2_seq: str,
    r2_qual: str,
) -> None:
    """Write one matched read's R1 and R2 records into its target index bucket.

    Looks ``tgidx`` up in ``writers`` before writing anything, so a failed lookup
    raises before either stream is touched -- there is no partial write. A
    ``writers`` dict built from chemistry's target index whitelist never contains
    ``"NONE"``, so an unmatched read routed here by mistake fails the same lookup as
    any other unknown target index, with no special casing required.

    Args:
        writers: Target index to its open (R1, R2) bucket streams.
        tgidx: The read's assigned target index, used as the lookup key.
        header: The already-rendered read name, written identically to both streams.
        r1_seq: The R1 sequence.
        r1_qual: The R1 quality string.
        r2_seq: The R2 sequence.
        r2_qual: The R2 quality string.

    Raises:
        KeyError: If ``tgidx`` has no open bucket in ``writers``, naming ``tgidx``.
    """
    if tgidx not in writers:
        raise KeyError(f"No open output bucket for target index '{tgidx}'")

    bucket = writers[tgidx]
    FastqFile.write_read(bucket.r1, header, r1_seq, r1_qual)
    FastqFile.write_read(bucket.r2, header, r2_seq, r2_qual)
