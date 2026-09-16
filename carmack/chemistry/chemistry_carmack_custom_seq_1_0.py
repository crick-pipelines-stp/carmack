"""
Carmack Custom Sequencing 1.0 chemistry definition.

Read structure (5' to 3'):
BC3 (10bp) -> PRIMER_C (22bp) -> BC2 (10bp) -> PRIMER_A (22bp) -> BC1 (10bp)
-> UMI (8bp) -> POLYG (homopolymer, min run 3) -> TGIDX (8bp) -> ME (19bp)

The UMI, poly-G and TGIDX components carry no known sequence, so barcode
matching and spacer checks ignore them; they model the post-barcode layout for
downstream UMI extraction only. ME's sequence is well known and is registered
on its ReadComponent (so it appears in read_structure.get_known_sequences()),
but verify=False keeps it out of spacer verification, for the reason given
where it is defined.
"""

from functools import cached_property
from importlib.resources import files

from carmack.chemistry.chemistry_base import (
    ChemistryBase,
    MatchErrors,
    WhitelistDistancePolicy,
    WhitelistSource,
)
from carmack.chemistry.chemistry_factory import ChemistryFactory
from carmack.chemistry.read_component import ReadComponent, ReadComponentType
from carmack.chemistry.read_structure import ReadStructure

# Primer sequences
PRIMER_C = "TGTGTATAAGGACCTCGTTGCC"
PRIMER_A = "ATGGAAGCCGACGAATTAGACC"

BC_CHUNK_LEN = 10

# UMI / poly-G anchor / TGIDX layout following BC1 (5' to 3').
UMI_LENGTH = 8
POLYG_BASE = "G"
POLYG_MIN_RUN = 3
TGIDX_LENGTH = 8

# Mosaic End, immediately 3' of the target index. Its sequence is known and registered below,
# but verify=False on its ReadComponent keeps MatcherBase.check_spacers from ever comparing it.
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

# Two BC2 entries one substitution apart. At a barcode budget of one there is no correction
# capacity left at that base, so a single A->G -- the dominant substitution direction on
# 2-colour Illumina chemistry -- turns one of these valid cell barcodes into the other,
# matching exactly at its expected position and reported as a perfect match. It is undetectable
# in code and only retiring an entry fixes it, which is a barcode-design decision about a plate
# well and about libraries already sequenced against the current set. Until that decision is
# taken the pair is declared here rather than left to fail construction, so the blind spot is
# recorded and warned about on every run instead of being silently absorbed.
BC2_INDISTINGUISHABLE_PAIR = frozenset({"AGCTTGAGAG", "GGCTTGAGAG"})

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
                sequence=ME,
                verify=False,
            ),
        ]

    @cached_property
    def max_errors(self) -> MatchErrors:
        """Return the maximum allowed errors for component matching."""
        return MatchErrors(barcode=1, spacer=2, tgidx=1)

    def whitelist_distance_policy(self) -> WhitelistDistancePolicy:
        """Enforce the whitelist distance bound, less one recorded exemption.

        This project designs these barcode sets, so it can retire an entry that breaks the
        bound and a violation is a defect rather than a fact of life. The single exemption is
        the BC2 pair described at BC2_INDISTINGUISHABLE_PAIR.

        Returns:
            An enforcing policy exempting only that pair.
        """
        return WhitelistDistancePolicy(
            enforce=True, exempt_pairs=frozenset({BC2_INDISTINGUISHABLE_PAIR})
        )

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
