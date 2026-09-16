"""Tests for the shared MultiQC report module.

``carmack.mqc_report`` carries the two identifiers every MultiQC custom
content section needs to attach itself to the Carmack parent module
(``CARMACK_PARENT_ID`` and ``CARMACK_PARENT_NAME``), plus the one place a
caller writes MQC-readable JSON payloads to disk. These tests pin the two
constants' exact values, since a MultiQC config keys off them literally.

Because those constants are carmack-wide rather than any one stage's, this is
also where the rule for using them is stated: which of the two keys a payload
must carry is decided by its plot type, and the assertion runs over every
payload all four stages render, reached by reflecting over their builders so a
payload added later is held to the same rule without this file being edited.
The four per-stage reporting test modules keep a one-line version of the same
check against their own generalstats payload, which fails closer to the stage.

``write_mqc_payloads`` is the writer that makes MultiQC's one-file-one-chart
rule structural instead of remembered. MultiQC opens a custom-content file,
looks for a top-level ``data`` key, and throws the whole file away with a
warning -- not an error -- when it does not find one, so a stage that bundles
several payloads under keys of its own loses every chart in that file
silently. These tests pin the properties that stop that recurring: each
payload is written alone, as the entire body of its own file; each filename is
derived from the payload's own ``id``, which is the unwritten convention the
already-rendering files obey and have to keep obeying byte for byte, because a
downstream consumer looks these files up by exact name; and a payload that
cannot name a file, or that carries nothing MultiQC would render, raises
rather than reaching disk. A ``None`` payload is skipped instead, since
returning ``None`` is how a builder says it has nothing to report and a run
with no matched read must not leave an empty chart behind. The one shape that
looks broken and is not is a payload whose ``data`` maps a sample to an empty
mapping: the top-level ``data`` is there, MultiQC renders it, and
assign-targets emits exactly that for a run in which nothing matched.
"""

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from assertpy import assert_that

from carmack.assign_targets.assign_reporting import AssignStats
from carmack.barcode.extraction_dataclasses import MatchMethod
from carmack.barcode.extraction_reporting import (
    ExtractionStats,
    OverallStats,
    PerBarcodeStats,
    to_mqc_barcode_rank,
)
from carmack.mqc_report import CARMACK_PARENT_ID, CARMACK_PARENT_NAME, write_mqc_payloads
from carmack.prepare_reads.prepare_reporting import PrepareStats
from carmack.umi.umi_reporting import UmiExtractionStats

SAMPLE_PREFIX = "SK609"

RENDERING_PAYLOAD_FILENAMES = [
    ("carmack_extraction_edit_distance", "SK609.extraction_edit_distance_mqc.json"),
    ("carmack_umi_anchor_run", "SK609.umi_anchor_run_mqc.json"),
    ("carmack_tgidx_edit_distance", "SK609.tgidx_edit_distance_mqc.json"),
    ("carmack_tgidx_anchor_run", "SK609.tgidx_anchor_run_mqc.json"),
]

GENERALSTATS_PLOT_TYPE = "generalstats"

# The two keys that attribute a payload to the Carmack parent, and the one key that
# does it for a generalstats table. Which pair applies is decided by the plot type.
PARENT_KEYS = ("parent_id", "parent_name")
NAMESPACE_KEY = "namespace"

# What the four stages render between them today: one generalstats table each, plus
# five bargraphs and five linegraphs. Counted so an assertion over the collected
# payloads cannot pass by having collected none of them.
CARMACK_GENERALSTATS_PAYLOAD_COUNT = 4
CARMACK_CHART_PAYLOAD_COUNT = 10


def renderable_payload(payload_id: str) -> dict[str, object]:
    """Build the smallest payload MultiQC renders: an id to name its file by, plus ``data``."""
    return {
        "id": payload_id,
        "plot_type": "linegraph",
        "parent_id": CARMACK_PARENT_ID,
        "parent_name": CARMACK_PARENT_NAME,
        "data": {SAMPLE_PREFIX: {"0": 1200, "1": 340, "2": 56}},
    }


def sibling_stats() -> tuple[ExtractionStats, UmiExtractionStats, AssignStats]:
    """Build one populated stats object for each carmack stage other than prepare-reads.

    Each is populated richly enough that none of its builders suppress their
    payload, so reflecting over these reaches every MultiQC payload those stages
    render rather than only the ones a degenerate run happens to produce.
    Prepare-reads is built separately by ``prepare_stats``, so that these three
    are also the id space prepare-reads' own ids have to stay clear of.

    Returns:
        The barcode, UMI and assign-targets stats objects.
    """
    return (
        ExtractionStats(
            overall=OverallStats(total_reads=1, perfect=1, corrok=0, fail=0, top_10_barcodes=[]),
            per_barcode=[
                PerBarcodeStats(
                    bc_name="BC1",
                    method=MatchMethod.EXACTMATCH,
                    attempts=1,
                    success=1,
                    fail=0,
                    edit_distance_dist=Counter({0: 1}),
                    reads_w_ambiguous_match=0,
                    spacer_present=0,
                )
            ],
            bc_names=["BC1"],
        ),
        UmiExtractionStats(
            total_reads=1,
            accepted=1,
            missing_left_anchor=0,
            truncated=0,
            umi_length=12,
            homopolymer_base="G",
            homopolymer_run_counts={4: 1},
        ),
        AssignStats(
            total_reads=1,
            matched=1,
            unmatched_no_match=0,
            unmatched_no_left_anchor_pos=0,
            unmatched_short_window=0,
            target_counts={"targetA": 1},
            edit_distance_counts={0: 1},
            homopolymer_base="G",
            homopolymer_run_counts={4: 1},
        ),
    )


def prepare_stats() -> PrepareStats:
    """Build the fourth stage's stats object, reconciling the way its producer must.

    Returns:
        A prepare-reads stats object whose unmatched count and per-target
        distribution sum to ``total_reads``, so neither of its builders is
        rendering a degenerate run.
    """
    return PrepareStats(
        total_reads=2, unmatched_written=1, target_written={"targetA": 1}, insert_not_sequenced=0
    )


def mqc_payloads(stats_objects: Sequence[object]) -> list[dict[str, Any]]:
    """Render every MultiQC payload the given stats objects build.

    The builders are found by reflection rather than named here, so a stage that
    renames or adds one is covered without this file being edited. A builder
    returning ``None`` measured nothing and renders no payload, so it contributes
    nothing.

    Args:
        stats_objects: Stats objects whose ``to_mqc_`` builders are called.

    Returns:
        Every payload rendered, in stats-object then builder-name order.
    """
    payloads: list[dict[str, Any]] = []
    for stats in stats_objects:
        for name in dir(stats):
            if not name.startswith("to_mqc_"):
                continue
            payload = getattr(stats, name)(SAMPLE_PREFIX)
            if payload is not None:
                payloads.append(payload)
    return payloads


def carmack_mqc_payloads() -> list[dict[str, Any]]:
    """Render every MultiQC payload carmack builds, over all four reporting stages.

    Most are reached by reflection. The barcode rank curve is appended by hand
    because reflection cannot reach it: it is a module-level function rather than
    a builder on a stats object, since the full barcode counts it plots are never
    carried on the finalized stats object. Wiring it in here holds it to the same
    attribution rules as every other chart.

    Returns:
        The payloads the barcode, UMI, assign-targets and prepare-reads stats
        objects render between them, plus the barcode rank curve.
    """
    return [
        *mqc_payloads([*sibling_stats(), prepare_stats()]),
        to_mqc_barcode_rank(SAMPLE_PREFIX, Counter({"ACGTACGTAC": 120, "TGCATGCATG": 45})),
    ]


def generalstats_mqc_payloads() -> list[dict[str, Any]]:
    """Return the carmack payloads MultiQC renders as General Statistics columns.

    Returns:
        Every payload whose plot type is generalstats.
    """
    return [p for p in carmack_mqc_payloads() if p["plot_type"] == GENERALSTATS_PLOT_TYPE]


def chart_mqc_payloads() -> list[dict[str, Any]]:
    """Return the carmack payloads MultiQC renders as a section of their own.

    Returns:
        Every payload whose plot type is something other than generalstats.
    """
    return [p for p in carmack_mqc_payloads() if p["plot_type"] != GENERALSTATS_PLOT_TYPE]


def mqc_payload_id(payload: dict[str, Any]) -> str:
    """Name a parametrized payload case after the MultiQC module id it declares.

    Args:
        payload: Payload the case runs against.

    Returns:
        The payload's module id, so a failure names the stage and chart at fault.
    """
    return str(payload["id"])


class TestCarmackParentConstants:
    """The two module-level identifiers a MultiQC config keys off literally."""

    def test_carmack_parent_id_is_carmack(self) -> None:
        """Test that CARMACK_PARENT_ID is exactly the lowercase module id."""
        assert_that(CARMACK_PARENT_ID).is_equal_to("carmack")

    def test_carmack_parent_name_is_carmack(self) -> None:
        """Test that CARMACK_PARENT_NAME is exactly the capitalised display name."""
        assert_that(CARMACK_PARENT_NAME).is_equal_to("Carmack")


class TestWriteMqcPayloadsFilenames:
    """``write_mqc_payloads``: the filename each payload earns from its own ``id``."""

    @pytest.mark.parametrize("payload_id,filename", RENDERING_PAYLOAD_FILENAMES)
    def test_write_mqc_payloads_reproduces_an_already_rendering_filename(
        self, payload_id: str, filename: str, tmp_path: Path
    ) -> None:
        """Test that each payload MultiQC already renders keeps its exact current filename.

        These four files are the ones a downstream pipeline reads by exact
        name, so deriving the name from the payload id is only safe while it
        reproduces them byte for byte. A mismatch here is a breaking output
        rename, not a cosmetic difference.
        """
        written = write_mqc_payloads(tmp_path, SAMPLE_PREFIX, [renderable_payload(payload_id)])

        assert_that(written).is_equal_to([tmp_path / filename])
        assert_that((tmp_path / filename).exists()).is_true()

    @pytest.mark.parametrize(
        "payload_id,stem",
        [
            ("prepare_stats", "prepare_stats"),
            ("carmack_carmack_stats", "carmack_stats"),
            ("tgidx_carmack_stats", "tgidx_carmack_stats"),
        ],
        ids=["no-carmack-prefix", "leading-prefix-stripped-once", "prefix-not-at-the-start"],
    )
    def test_write_mqc_payloads_strips_only_a_leading_carmack_prefix(
        self, payload_id: str, stem: str, tmp_path: Path
    ) -> None:
        """Test that an id without a leading ``carmack_`` passes through unaltered.

        The strip has to be a no-op wherever the prefix is absent or occurs
        later in the id: a naive replace would silently corrupt the section
        name, and the section name is also the id MultiQC anchors the report on.
        """
        written = write_mqc_payloads(tmp_path, SAMPLE_PREFIX, [renderable_payload(payload_id)])

        assert_that(written).is_equal_to([tmp_path / f"{SAMPLE_PREFIX}.{stem}_mqc.json"])

    def test_write_mqc_payloads_names_every_file_with_the_given_prefix(
        self, tmp_path: Path
    ) -> None:
        """Test that the sample prefix, not a hardcoded one, leads every filename."""
        written = write_mqc_payloads(
            tmp_path, "SK661", [renderable_payload("carmack_umi_anchor_run")]
        )

        assert_that(written).is_equal_to([tmp_path / "SK661.umi_anchor_run_mqc.json"])

    def test_write_mqc_payloads_writes_inside_the_given_output_directory(
        self, tmp_path: Path
    ) -> None:
        """Test that the file lands in the directory passed in, not the working directory."""
        output_dir = tmp_path / "reports"
        output_dir.mkdir()

        written = write_mqc_payloads(
            output_dir, SAMPLE_PREFIX, [renderable_payload("carmack_umi_anchor_run")]
        )

        assert_that(written[0].parent).is_equal_to(output_dir)
        assert_that(list(tmp_path.glob("*.json"))).is_empty()


class TestWriteMqcPayloadsFileContents:
    """``write_mqc_payloads``: what ends up inside each file it writes."""

    def test_write_mqc_payloads_writes_the_payload_as_the_whole_file_body(
        self, tmp_path: Path
    ) -> None:
        """Test that the payload is the entire file, with no enclosing key around it.

        A wrapper is precisely what MultiQC discards: it looks for ``data`` at
        the top level and drops the file when a stage's own key hides it.
        """
        payload = renderable_payload("carmack_tgidx_edit_distance")

        written = write_mqc_payloads(tmp_path, SAMPLE_PREFIX, [payload])

        with open(written[0]) as handle:
            loaded = json.load(handle)
        assert_that(loaded).is_equal_to(payload)

    def test_write_mqc_payloads_serialises_with_two_space_indentation(
        self, tmp_path: Path
    ) -> None:
        """Test that files stay two-space indented, so a diff of a report output stays readable."""
        payload = renderable_payload("carmack_umi_anchor_run")

        written = write_mqc_payloads(tmp_path, SAMPLE_PREFIX, [payload])

        assert_that(written[0].read_text()).is_equal_to(json.dumps(payload, indent=2))

    def test_write_mqc_payloads_gives_every_payload_a_file_of_its_own(
        self, tmp_path: Path
    ) -> None:
        """Test that several payloads in one call land in separate files, none overwriting another.

        One call per stage writing all of that stage's charts is the shape
        that replaces the bundled file, so the split has to happen inside the
        writer rather than at each call site.
        """
        payloads = [
            renderable_payload(payload_id) for payload_id, _ in RENDERING_PAYLOAD_FILENAMES
        ]

        written = write_mqc_payloads(tmp_path, SAMPLE_PREFIX, payloads)

        assert_that(written).is_equal_to(
            [tmp_path / filename for _, filename in RENDERING_PAYLOAD_FILENAMES]
        )
        for path, payload in zip(written, payloads):
            with open(path) as handle:
                assert_that(json.load(handle)).is_equal_to(payload)

    def test_write_mqc_payloads_returns_paths_that_exist_on_disk(self, tmp_path: Path) -> None:
        """Test that every returned path is a real file, so a caller can count what it emitted."""
        payloads = [
            renderable_payload(payload_id) for payload_id, _ in RENDERING_PAYLOAD_FILENAMES
        ]

        written = write_mqc_payloads(tmp_path, SAMPLE_PREFIX, payloads)

        assert_that(written).is_length(4)
        for path in written:
            assert_that(path.exists()).is_true()

    def test_write_mqc_payloads_accepts_a_single_pass_iterable(self, tmp_path: Path) -> None:
        """Test that a generator is consumed correctly, since builders are chained lazily."""
        payloads = (
            renderable_payload(payload_id) for payload_id, _ in RENDERING_PAYLOAD_FILENAMES
        )

        written = write_mqc_payloads(tmp_path, SAMPLE_PREFIX, payloads)

        assert_that(written).is_equal_to(
            [tmp_path / filename for _, filename in RENDERING_PAYLOAD_FILENAMES]
        )


class TestWriteMqcPayloadsConditionalEmission:
    """``write_mqc_payloads``: a builder returning ``None`` means "nothing to report"."""

    def test_write_mqc_payloads_skips_a_none_payload(self, tmp_path: Path) -> None:
        """Test that a ``None`` payload writes no file at all.

        Builders return ``None`` when they measured nothing, and an empty
        chart file is worse than an absent one: MultiQC would render a section
        stating a distribution that was never observed.
        """
        written = write_mqc_payloads(tmp_path, SAMPLE_PREFIX, [None])

        assert_that(written).is_empty()
        assert_that(list(tmp_path.iterdir())).is_empty()

    def test_write_mqc_payloads_writes_only_the_non_none_payloads_of_a_mixed_sequence(
        self, tmp_path: Path
    ) -> None:
        """Test that a mix of payloads and ``None`` writes and returns only the real payloads.

        This is the shape a stage actually passes: some charts are always
        emitted and others are conditional on having measured anything.
        """
        edit_distance = renderable_payload("carmack_tgidx_edit_distance")
        anchor_run = renderable_payload("carmack_tgidx_anchor_run")

        written = write_mqc_payloads(
            tmp_path, SAMPLE_PREFIX, [edit_distance, None, anchor_run, None]
        )

        assert_that(written).is_equal_to(
            [
                tmp_path / "SK609.tgidx_edit_distance_mqc.json",
                tmp_path / "SK609.tgidx_anchor_run_mqc.json",
            ]
        )
        assert_that(sorted(path.name for path in tmp_path.iterdir())).is_equal_to(
            ["SK609.tgidx_anchor_run_mqc.json", "SK609.tgidx_edit_distance_mqc.json"]
        )

    def test_write_mqc_payloads_writes_nothing_for_an_empty_iterable(self, tmp_path: Path) -> None:
        """Test that a stage with no payloads at all is a no-op rather than an error."""
        written = write_mqc_payloads(tmp_path, SAMPLE_PREFIX, [])

        assert_that(written).is_empty()
        assert_that(list(tmp_path.iterdir())).is_empty()


class TestWriteMqcPayloadsRejectsUnrenderablePayloads:
    """``write_mqc_payloads``: refusing the shapes MultiQC would silently discard.

    ``ValueError`` is the fit: nothing is missing from the filesystem and no
    type contract is broken -- the payload is simply a value this writer
    cannot turn into a file MultiQC will read. Raising rather than warning is
    the point, since the defect being fixed here was invisible precisely
    because MultiQC only warns.
    """

    @pytest.mark.parametrize(
        "payload",
        [
            {"plot_type": "linegraph", "data": {SAMPLE_PREFIX: {"0": 1}}},
            {"id": "", "data": {SAMPLE_PREFIX: {"0": 1}}},
            {"id": None, "data": {SAMPLE_PREFIX: {"0": 1}}},
            {"id": 17, "data": {SAMPLE_PREFIX: {"0": 1}}},
        ],
        ids=["no-id-key", "empty-string", "none", "not-a-string"],
    )
    def test_write_mqc_payloads_rejects_a_payload_that_cannot_name_its_file(
        self, payload: dict[str, object], tmp_path: Path
    ) -> None:
        """Test that a missing, empty or non-string id raises instead of writing a nameless file.

        The filename is derived from the id, so a payload without a usable one
        has no file to be written to and no stable section id in the report.
        """
        with pytest.raises(ValueError) as error:
            write_mqc_payloads(tmp_path, SAMPLE_PREFIX, [payload])

        assert_that(str(error.value)).contains("id")
        assert_that(list(tmp_path.iterdir())).is_empty()

    @pytest.mark.parametrize(
        "payload",
        [
            {"id": "carmack_tgidx_edit_distance", "plot_type": "linegraph"},
            {"id": "carmack_tgidx_edit_distance", "data": {}},
            {"id": "carmack_tgidx_edit_distance", "data": None},
            {"id": "carmack_tgidx_edit_distance", "data": []},
        ],
        ids=["no-data-key", "empty-mapping", "none", "empty-sequence"],
    )
    def test_write_mqc_payloads_rejects_a_payload_with_no_top_level_data(
        self, payload: dict[str, object], tmp_path: Path
    ) -> None:
        """Test that a payload MultiQC would discard raises, naming the payload at fault.

        A missing or empty top-level ``data`` is exactly the shape MultiQC
        drops with a warning, so the writer refuses it here where the failure
        is loud and attributable to one payload.
        """
        with pytest.raises(ValueError) as error:
            write_mqc_payloads(tmp_path, SAMPLE_PREFIX, [payload])

        assert_that(str(error.value)).contains("carmack_tgidx_edit_distance")
        assert_that(list(tmp_path.iterdir())).is_empty()

    def test_write_mqc_payloads_writes_nothing_when_a_later_payload_is_rejected(
        self, tmp_path: Path
    ) -> None:
        """Test that a good payload ahead of a rejected one still leaves the directory empty.

        Checking each payload as its file went out would leave everything before
        the bad one on disk, so a raising run would still publish a partial set of
        files -- and a partial set is exactly what MultiQC would read as the whole
        report. Validating the whole sequence first makes the write all-or-nothing.
        """
        good = renderable_payload("carmack_tgidx_edit_distance")
        bad = {"id": "carmack_tgidx_anchor_run", "plot_type": "linegraph", "data": {}}

        with pytest.raises(ValueError) as error:
            write_mqc_payloads(tmp_path, SAMPLE_PREFIX, [good, bad])

        assert_that(str(error.value)).contains("carmack_tgidx_anchor_run")
        assert_that(list(tmp_path.iterdir())).is_empty()

    def test_write_mqc_payloads_writes_a_payload_whose_per_sample_data_is_empty(
        self, tmp_path: Path
    ) -> None:
        """Test that ``data`` mapping a sample to an empty mapping is valid and is written.

        Only the top level is checked for emptiness. assign-targets emits
        ``{"SK609": {}}`` for a run in which no read matched: the top-level
        ``data`` key is present, so MultiQC renders the section, and refusing
        it would turn a legitimate empty result into a hard failure.
        """
        payload = {
            "id": "carmack_tgidx_edit_distance",
            "plot_type": "linegraph",
            "data": {SAMPLE_PREFIX: {}},
        }

        written = write_mqc_payloads(tmp_path, SAMPLE_PREFIX, [payload])

        assert_that(written).is_equal_to([tmp_path / "SK609.tgidx_edit_distance_mqc.json"])
        with open(written[0]) as handle:
            assert_that(json.load(handle)).is_equal_to(payload)


class TestCarmackMqcPayloadAttribution:
    """How every carmack payload, at every stage, attributes itself to the Carmack parent.

    Two different keys do that job, and a payload's plot type decides which one
    applies. MultiQC's custom-content parser branches on the generalstats plot type
    and returns before the point where ``parent_id`` is read, so on a generalstats
    payload the parent keys are inert: the columns come out attributed to the raw
    payload id, as ``custom_content_carmack_extraction_general_stats-pct_perfect``
    rather than as one Carmack column among four stages'. The key that does control
    that attribution is ``namespace``, which falls back to the module id when unset.

    Every other plot type takes the branch that does read ``parent_id``, and there
    the parent pair is live config: it is what nests the five bargraphs and five
    linegraphs as sibling sections under one Carmack heading. The two shapes are
    therefore not interchangeable, and tidying the charts into the generalstats
    shape would scatter their sections without any error being raised.

    The rule is about the two constants this module exposes, so it is stated here
    once over every stage rather than four times over one, and reached by
    reflecting over the builders, so a payload added later is held to the same rule
    without this file being edited.
    """

    @pytest.mark.parametrize("payload", generalstats_mqc_payloads(), ids=mqc_payload_id)
    def test_generalstats_payload_declares_the_carmack_namespace(
        self, payload: dict[str, Any]
    ) -> None:
        """Test that every stage's General Statistics columns are attributed to Carmack."""
        assert_that(payload).contains_entry({NAMESPACE_KEY: CARMACK_PARENT_NAME})

    @pytest.mark.parametrize("payload", generalstats_mqc_payloads(), ids=mqc_payload_id)
    def test_generalstats_payload_carries_no_parent_keys(self, payload: dict[str, Any]) -> None:
        """Test that no generalstats payload keeps parent keys its branch of the parser ignores."""
        assert_that(payload).does_not_contain_key(*PARENT_KEYS)

    @pytest.mark.parametrize("payload", chart_mqc_payloads(), ids=mqc_payload_id)
    def test_chart_payload_carries_the_shared_carmack_parent_identifiers(
        self, payload: dict[str, Any]
    ) -> None:
        """Test that every bargraph and linegraph still nests under the shared Carmack parent."""
        assert_that(payload).contains_entry(
            {"parent_id": CARMACK_PARENT_ID}, {"parent_name": CARMACK_PARENT_NAME}
        )

    @pytest.mark.parametrize("payload", chart_mqc_payloads(), ids=mqc_payload_id)
    def test_chart_payload_declares_no_namespace(self, payload: dict[str, Any]) -> None:
        """Test that no chart payload takes on the generalstats shape, which would not nest it."""
        assert_that(payload).does_not_contain_key(NAMESPACE_KEY)

    def test_payload_collection_reaches_every_payload_the_four_stages_render(self) -> None:
        """Test that the attribution rules above are stated over payloads that were found.

        They are parametrized over the collected payloads, so a collection that
        reached none of them would leave every one of those cases passing
        vacuously. This counts what was reached: one General Statistics table per
        stage, and the ten charts between them. The collection is no longer purely
        reflective -- the barcode rank curve is a module-level builder reflection
        cannot see, and is wired in by hand -- so this count is also what catches a
        chart that stops being collected at all.
        """
        assert_that(generalstats_mqc_payloads()).is_length(CARMACK_GENERALSTATS_PAYLOAD_COUNT)
        assert_that(chart_mqc_payloads()).is_length(CARMACK_CHART_PAYLOAD_COUNT)
