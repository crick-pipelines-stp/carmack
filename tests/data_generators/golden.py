# Provenance of the golden regression inputs in tests/data/golden/
#
# These FASTQ files are the fixed inputs to the golden output baseline. They are
# committed so that every later change to the barcode matchers can be measured
# against a known-good set of outputs.
#
#   custom_seq_1_0_R1.fastq.gz  - 2000 reads of a real carmack_custom_seq_1_0
#                                 library (SK661). Read layout:
#                                 BC3(10) PRIMER_C(22) BC2(10) PRIMER_A(22)
#                                 BC1(10) UMI(8) poly-G TGIDX(8) Tn5 insert
#
#   hydrop_R1.fastq.gz          - 2000 reads taken from the head of the
#                                 committed hydrop library, subset for speed:
#
#                                 zcat tests/data/hydrop_scatac_1_S1_R1_001.fastq.gz \
#                                     | head -8000 | gzip -n > hydrop_R1.fastq.gz
#
# The two "small" inputs are the always-run tier. They are strict head-subsets of
# the full inputs above, chosen to keep the default test suite fast while still
# driving every matcher tier (EXACTMATCH, KMERMATCH and ALIGNMATCH all appear for
# all three barcode components in the resulting bc_stats.txt report):
#
#   zcat custom_seq_1_0_R1.fastq.gz | head -800 | gzip -n > custom_seq_1_0_small_R1.fastq.gz
#   zcat hydrop_R1.fastq.gz         | head -200 | gzip -n > hydrop_small_R1.fastq.gz
#
# gzip -n omits the timestamp so re-deriving the inputs reproduces identical bytes.
#
# custom_seq_1_0_small_R2.fastq.gz - synthetic R2 mate for custom_seq_1_0_small_R1.fastq.gz
# custom_seq_1_0_R2.fastq.gz       - synthetic R2 mate for custom_seq_1_0_R1.fastq.gz
#
#   No real R2 was ever collected alongside either committed custom_seq_1_0 R1 input, so
#   ReadPreparer -- which requires a genuine paired R1/R2 FASTQ -- has no real mate to run
#   against. These two files exist to close that gap without fabricating biological
#   signal: ReadPreparer, ScrnaWriter and the scTIP writer never read R2 content for any
#   computation, only pass it through untouched, so a synthesized sequence is a legitimate
#   stand-in here, not a fabrication of anything the pipeline actually inspects.
#
#   Each file carries exactly the same number of records, in the same order, as its R1
#   counterpart, with every record's read id (the header's first whitespace token) copied
#   verbatim from the corresponding R1 record -- extracted by reading every 4th line of
#   the real committed R1 gzip, starting at line 0, stripping the leading "@" and taking
#   the first whitespace-delimited token. This mirrors a real sequencer's R1/R2 output,
#   where every physical read has a mate regardless of what a later software pipeline
#   decides to keep: it is the target-annotated R1 (barcode- and UMI-extraction survivors
#   only, a strict, order-preserving subsequence of the full input) that the golden test
#   harness then pairs a read down to before ever handing it to ReadPreparer -- see
#   tests/test_golden_outputs.py's `build_prepare_r2` helper.
#
#   Each record's sequence and quality are derived deterministically from its read id
#   alone, so regenerating the files reproduces identical bytes:
#
#     - Sequence: SHA-256 the read id, map each digest byte to "ACGT"[byte % 4], and
#       repeat/truncate the digest to a fixed 50 bases.
#     - Quality: SHA-256 the read id with a ":qual" suffix (so it differs from the
#       sequence's own digest), and map each byte to the Phred+33 character
#       chr(33 + byte % 40) -- the same "vary by position, stay in a sane printable
#       range" scheme tests/test_read_preparer.py's own `make_qual` helper uses,
#       applied per read instead of per position so distinct reads are still
#       distinguishable by eye.
#     - Header: "{read_id} 2:N:0:1", an arbitrary but plausible second token -- only the
#       first token is ever read back as the id.
#
#   Compressed exactly like tests/utils.py's `gzip_bytes` helper does: written into a
#   gzip.GzipFile opened over an in-memory buffer with mtime=0 and no filename embedded,
#   which is the Python equivalent of piping through `gzip -n` and reproduces identical
#   bytes on every regeneration, on any machine.
#
# prepare_reads_multi_target_R1.fastq.gz - hand-built, multi-target ReadPreparer fixture
# prepare_reads_multi_target_R2.fastq.gz - its paired R2
#
#   The real carmack_custom_seq_1_0 chemistry's target index whitelist carries exactly one
#   confirmed entry, so no fixture built from a real library could ever exercise more than
#   one scTIP target bucket. These two files are a deliberately synthetic fixture built to
#   close that gap: 22 read pairs, already shaped exactly like a real assign-targets run's
#   own output (carrying BC1/BC2/BC3(+_POS), UMI/UMI_POS, and TGIDX(+TGIDX_POS when matched)
#   header tags), split across the unmatched arm and two matched target buckets from the
#   synthetic ChemistryTwoTargets whitelist ("TATAGCCT", "CATTGGAC") that
#   tests/test_read_preparer.py already defines and reuses throughout its own unit tests.
#   No claim is made about how a real multi-target library would behave; the fixture exists
#   only to exercise ReadPreparer's dispatch across more than one matched bucket end to end.
#
#   Composition (22 read pairs total):
#     - 3 unmatched reads carrying no TGIDX tag at all -- the shape a chemistry with no
#       target index support emits (ids: unmatched_no_tag_1..3).
#     - 3 unmatched reads instead carrying an explicit TGIDX=NONE tag with no TGIDX_POS
#       span -- the shape a real assign-targets run emits for an unmatched read on a
#       chemistry that DOES support target assignment, since `TargetAssigner.assign_read`
#       never writes a span for the unassigned sentinel (ids: unmatched_none_1..3).
#     - 8 reads matched to the first whitelist entry, TATAGCCT (ids: matched_a_1..8).
#     - 8 reads matched to the second whitelist entry, CATTGGAC (ids: matched_b_1..8).
#
#   Every read shares the same barcode/UMI tag values and the same read-structure layout
#   (BC3(10) BC2(10) BC1(10) UMI(8) POLYG-run TGIDX(8) ME-length filler insert) that
#   tests/test_read_preparer.py's `build_full_r1_read`/`make_prepare_records` helpers
#   already build for the rest of that file's whole-run tests -- the fixture generation
#   recipe reuses those helpers directly, plus the module's own
#   `MULTI_TARGET_IDS_AND_TARGETS`/`MULTI_TARGET_EXPLICIT_NONE_IDS` constants, rather than
#   reimplementing equivalent read-building logic, so the committed fixture and the test
#   class asserting against it (`TestReadPreparerMultiTargetGoldenEndToEnd`) cannot drift
#   apart. R2 is synthesized exactly as described above for the custom_seq_1_0 R2 fixtures
#   (SHA-256-derived ACGT sequence and Phred+33 quality per read id), and both files are
#   compressed the same gzip -n equivalent way.
