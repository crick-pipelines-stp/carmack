"""Tests for the shared MultiQC report module.

``carmack.mqc_report`` carries the two identifiers every MultiQC custom
content section needs to attach itself to the Carmack parent module
(``CARMACK_PARENT_ID`` and ``CARMACK_PARENT_NAME``), plus ``write_mqc_json``,
the one place a caller writes an MQC-readable JSON payload to disk. These
tests pin the two constants' exact values, since a MultiQC config keys off
them literally, and pin ``write_mqc_json`` on the property that matters to a
downstream MultiQC parse: a payload written out and read back with
``json.load`` reproduces exactly what was passed in, for both a flat mapping
and one nesting a mapping inside it, the shape the generalstats and bargraph
payloads this module serializes elsewhere will actually take.

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
from pathlib import Path

import pytest
from assertpy import assert_that

from carmack.mqc_report import (
    CARMACK_PARENT_ID,
    CARMACK_PARENT_NAME,
    write_mqc_json,
    write_mqc_payloads,
)

FLAT_PAYLOAD = {"sample": "SK588", "total_reads": 12345, "matched_fraction": 0.875}

NESTED_PAYLOAD = {
    "id": "carmack_prepare_reads",
    "data": {
        "SK588": {"total_reads": 12345, "unmatched_written": 4321},
        "SK661": {"total_reads": 6789, "unmatched_written": 1234},
    },
    "config": {"title": "Prepare Reads", "sort_rows": False},
}

SAMPLE_PREFIX = "SK609"

RENDERING_PAYLOAD_FILENAMES = [
    ("carmack_extraction_edit_distance", "SK609.extraction_edit_distance_mqc.json"),
    ("carmack_umi_anchor_run", "SK609.umi_anchor_run_mqc.json"),
    ("carmack_tgidx_edit_distance", "SK609.tgidx_edit_distance_mqc.json"),
    ("carmack_tgidx_anchor_run", "SK609.tgidx_anchor_run_mqc.json"),
]


def renderable_payload(payload_id: str) -> dict[str, object]:
    """Build the smallest payload MultiQC renders: an id to name its file by, plus ``data``."""
    return {
        "id": payload_id,
        "plot_type": "linegraph",
        "parent_id": CARMACK_PARENT_ID,
        "parent_name": CARMACK_PARENT_NAME,
        "data": {SAMPLE_PREFIX: {"0": 1200, "1": 340, "2": 56}},
    }


class TestCarmackParentConstants:
    """The two module-level identifiers a MultiQC config keys off literally."""

    def test_carmack_parent_id_is_carmack(self) -> None:
        """Test that CARMACK_PARENT_ID is exactly the lowercase module id."""
        assert_that(CARMACK_PARENT_ID).is_equal_to("carmack")

    def test_carmack_parent_name_is_carmack(self) -> None:
        """Test that CARMACK_PARENT_NAME is exactly the capitalised display name."""
        assert_that(CARMACK_PARENT_NAME).is_equal_to("Carmack")


class TestWriteMqcJsonRoundTrip:
    """``write_mqc_json``: round-tripping a payload through disk via json.load."""

    def test_write_mqc_json_round_trips_a_flat_payload(self, tmp_path: Path) -> None:
        """Test that a flat dict payload reads back exactly as it was written."""
        output_path = tmp_path / "flat_mqc.json"

        write_mqc_json(output_path, FLAT_PAYLOAD)

        with open(output_path) as handle:
            loaded = json.load(handle)
        assert_that(loaded).is_equal_to(FLAT_PAYLOAD)

    def test_write_mqc_json_round_trips_a_nested_payload(self, tmp_path: Path) -> None:
        """Test that a dict nesting further dicts reads back exactly as it was written.

        This mirrors the generalstats and bargraph payload shapes the module
        serializes elsewhere, where the top-level mapping's values are
        themselves mappings.
        """
        output_path = tmp_path / "nested_mqc.json"

        write_mqc_json(output_path, NESTED_PAYLOAD)

        with open(output_path) as handle:
            loaded = json.load(handle)
        assert_that(loaded).is_equal_to(NESTED_PAYLOAD)

    def test_write_mqc_json_creates_a_readable_file_at_the_given_path(
        self, tmp_path: Path
    ) -> None:
        """Test that the file is written at exactly the path given, not some other location."""
        output_path = tmp_path / "subdir_check" / "report_mqc.json"
        output_path.parent.mkdir()

        write_mqc_json(output_path, FLAT_PAYLOAD)

        assert_that(output_path.exists()).is_true()


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

    def test_write_mqc_payloads_serialises_with_the_same_indentation_as_write_mqc_json(
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
