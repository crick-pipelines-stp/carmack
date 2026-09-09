"""
Carmack Custom Sequencing 1.0 chemistry definition.

Read structure (5' to 3'):
BC3 (10bp) -> PRIMER_C (22bp) -> BC2 (10bp) -> PRIMER_A (22bp) -> BC1 (10bp)
-> UMI (8bp, +/-1) -> POLYG (homopolymer, min run 3) -> TGIDX (8bp) -> ME (19bp)

The UMI, poly-G and TGIDX components carry no known sequence, so barcode
matching and spacer checks ignore them; they model the post-barcode layout for
downstream UMI extraction only. ME's sequence is well known (see the ME
constant below), but is deliberately left off its ReadComponent, so the same
matching and spacer checks ignore it too, for the different reason explained
where it is defined: ME anchors the target index's right edge for downstream
insert-boundary arithmetic, not UMI extraction.
"""

from functools import cached_property
from importlib.resources import files

from carmack.chemistry.chemistry_base import ChemistryBase, MatchErrors, WhitelistSource
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.chemistry.read_component import ReadComponent, ReadComponentType
from carmack.chemistry.read_structure import ReadStructure

# Primer sequences
PRIMER_C = "TGTGTATAAGGACCTCGTTGCC"
PRIMER_A = "ATGGAAGCCGACGAATTAGACC"

BC_CHUNK_LEN = 10

# UMI / poly-G anchor / TGIDX layout following BC1 (5' to 3').
UMI_LENGTH = 8
UMI_LENGTH_TOLERANCE = 1
POLYG_BASE = "G"
POLYG_MIN_RUN = 3
TGIDX_LENGTH = 8

# Mosaic End, immediately 3' of the target index. Used only for its length by the
# insert-boundary arithmetic elsewhere in the pipeline; no matcher is ever built over it, and
# its component below deliberately omits `sequence` so it can never be matched even
# incidentally. Two independent mechanisms protect this:
# FixedPositionMatcher is the only matcher that reads .start unguarded, and
# MatcherBase.__init__ structurally rejects any component whose type is not in
# allowed_component_types (default {BARCODE}), so a PRIMER-typed component like this one can
# never be constructed into a matcher directly. BarcodeExtractor.init_matchers() also explicitly
# skips every non-BARCODE component before building matchers at all.
# Separately, KmerMatcher widens allowed_component_types to include TGIDX, so TargetAssigner
# builds a KmerMatcher directly over TGIDX; when that matcher hits a same-score tie, its
# inherited check_spacers()/match_seq() looks at ReadStructure.get_next(TGIDX), which is this
# ME component, and only performs a real comparison when the neighbour's `sequence` is truthy.
# Leaving `sequence` unset on ME's component keeps that guard structurally false, so ME is never
# read as a spacer sequence either, even though it now sits immediately after TGIDX.
ME = "AGATGTGTATAAGAGACAG"

# Barcode file paths
BC1_PATH = files("carmack.data.barcodes.carmack.custom_seq").joinpath(
    "carmack_custom_seq_1_0_96_bc1.tsv"
)
BC2_PATH = files("carmack.data.barcodes.carmack.custom_seq").joinpath(
    "carmack_custom_seq_1_0_96_bc2.tsv"
)
BC3_PATH = files("carmack.data.barcodes.carmack.custom_seq").joinpath(
    "carmack_custom_seq_1_0_96_bc3.tsv"
)

# The confirmed target indexes ship as data, one sequence per line, so the set can grow
# without a code change. The loader has no comment syntax, so the file carries sequences only.
TGIDX_PATH = files("carmack.data.tgidx.carmack.custom_seq").joinpath(
    "carmack_custom_seq_1_0_tgidx.tsv"
)


class ChemistryCarmackCustomSeq10(ChemistryBase):
    """
    Chemistry definition for Carmack Custom Sequencing 1.0.

    This chemistry has a specific read structure with three barcode components
    and two primer sequences. Barcode whitelists are loaded from TSV files.
    """

    @cached_property
    def name(self) -> str:
        """Return the unique identifier for this chemistry."""
        return "carmack_custom_seq_1_0"

    @cached_property
    def read_structure(self) -> ReadStructure:
        """Define the layout of barcodes and primers within the read."""
        return ReadStructure(self._build_components())

    def _build_components(self) -> list[ReadComponent]:
        return [
            ReadComponent(name="BC3", type=ReadComponentType.BARCODE, length=BC_CHUNK_LEN),
            ReadComponent(
                name="PRIMER_C",
                type=ReadComponentType.PRIMER,
                length=len(PRIMER_C),
                sequence=PRIMER_C,
            ),
            ReadComponent(name="BC2", type=ReadComponentType.BARCODE, length=BC_CHUNK_LEN),
            ReadComponent(
                name="PRIMER_A",
                type=ReadComponentType.PRIMER,
                length=len(PRIMER_A),
                sequence=PRIMER_A,
            ),
            ReadComponent(name="BC1", type=ReadComponentType.BARCODE, length=BC_CHUNK_LEN),
            ReadComponent(
                name="UMI",
                type=ReadComponentType.UMI,
                length=UMI_LENGTH,
                length_tolerance=UMI_LENGTH_TOLERANCE,
            ),
            ReadComponent(
                name="POLYG",
                type=ReadComponentType.HOMOPOLYMER,
                homopolymer_base=POLYG_BASE,
                min_run=POLYG_MIN_RUN,
            ),
            ReadComponent(name="TGIDX", type=ReadComponentType.TGIDX, length=TGIDX_LENGTH),
            ReadComponent(
                name="ME",
                type=ReadComponentType.PRIMER,
                length=len(ME),
            ),
        ]

    @cached_property
    def max_errors(self) -> MatchErrors:
        """Return the maximum allowed errors for component matching."""
        return MatchErrors(barcode=1, spacer=2, tgidx=1)

    def whitelist_sources(self) -> dict[str, WhitelistSource]:
        """Return the packaged whitelist file for each whitelisted component.

        Returns:
            Dictionary mapping component names to their whitelist sources. Each
            line of these files is one sequence on its own, for the three
            barcodes and the target index alike.
        """
        return {
            "BC1": WhitelistSource(path=BC1_PATH),
            "BC2": WhitelistSource(path=BC2_PATH),
            "BC3": WhitelistSource(path=BC3_PATH),
            "TGIDX": WhitelistSource(path=TGIDX_PATH),
        }


# Register this chemistry with the factory
ChemistryFactory.register(ChemistryCarmackCustomSeq10)
