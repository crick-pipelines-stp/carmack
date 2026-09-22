# pylint: disable=missing-function-docstring, missing-class-docstring

import tempfile
import unittest
from unittest import mock

from click.testing import CliRunner

import carmack.__main__

R1_PATH = "tests/data/hydrop_scatac_1_S1_R1_001.fastq.gz"
CB_PATH = "tests/data/hydrop_scatac_1_S1_R2_001.fastq.gz"
BAM_PATH = "tests/data/hydrop_scatac_1_S1_R1.sorted.bam"
BAI_PATH = "tests/data/hydrop_scatac_1_S1_R1.sorted.bam.bai"
TAGGED_BAM_PATH = "tests/data/hydrop_scatac_1_S1_R1.dedup.tagged.bam"
BED_PATH = "tests/data/atac_k562_peaks.sorted.bed"
BC_VALID_PATH = "tests/data/bc_valid.csv"
LINEAR_DEDUP_BAM_PATH = "tests/data/linear_dedup_fixture.bam"
LINEAR_DEDUP_BAI_PATH = "tests/data/linear_dedup_fixture.bam.bai"


@mock.patch("carmack.__main__.carmack_cli")
def test_header(mock_cli, capsys):
    """Test running the header function"""

    carmack.__main__.run_carmack()
    out, err = capsys.readouterr()
    # print(err)
    assert "carmack" in err
    assert carmack.__version__ in err


class TestCli(unittest.TestCase):
    """Class for testing the command line interface"""

    def setUp(self):
        self.runner = CliRunner()
        self.tmp_dir = tempfile.mkdtemp()

    def assemble_params(self, params):
        """Assemble a dictionary of parameters into a list of arguments for the cli"""

        arg_list = []
        for key, value in params.items():
            if value is not None:
                arg_list += [f"--{key}", value]
            else:
                arg_list += [f"--{key}"]

        return arg_list

    def invoke_cli(self, cmd):
        """Invoke the commandline interface using a list of parameters"""

        return self.runner.invoke(carmack.__main__.carmack_cli, cmd)

    def test_cli_command_help(self):
        """Test the main launch function with --help"""

        result = self.invoke_cli(["--help"])
        # print(result)
        assert result.exit_code == 0
        assert "Show this message and exit." in result.output

    def test_cli_command_incorrect(self):
        """Test the main launch function with an unrecognised subcommand"""

        result = self.invoke_cli(["foo"])
        # print(result)
        self.assertTrue(result.exit_code == 2)

    @mock.patch("carmack.__main__.BarcodeExtractor", autospec=True)
    def test_cli_command_barcode_extractor(self, mock_barcode_extractor):
        """Test barcode_extractor"""

        # Init
        params = {"chemistry": "hydrop", "output_dir": ".", "prefix": "", "cpu_count": "1"}

        # Test
        cmd = ["extract-barcodes"] + [R1_PATH] + self.assemble_params(params)
        result = self.invoke_cli(cmd)

        # Assert
        self.assertTrue(result.exit_code == 0)
        mock_barcode_extractor.assert_called_once_with(R1_PATH, "hydrop", n_workers=1, fast=False)

    @mock.patch("carmack.__main__.BarcodeExtractor", autospec=True)
    def test_cli_command_barcode_extractor_fast_mode(self, mock_barcode_extractor):
        """Test barcode_extractor with fast mode enabled."""

        params = {
            "chemistry": "hydrop",
            "output_dir": ".",
            "prefix": "",
            "cpu_count": "1",
            "fast": None,
        }

        cmd = ["extract-barcodes"] + [R1_PATH] + self.assemble_params(params)
        result = self.invoke_cli(cmd)

        self.assertTrue(result.exit_code == 0)
        mock_barcode_extractor.assert_called_once_with(R1_PATH, "hydrop", n_workers=1, fast=True)

    @mock.patch("carmack.__main__.time.perf_counter")
    @mock.patch("carmack.__main__.atexit.register")
    @mock.patch("carmack.__main__.format_duration")
    def test_cli_time_logging(self, mock_format_duration, mock_atexit_register, mock_perf_counter):
        """Test that time logging is set up correctly in run_carmack"""

        # Setup mock return values
        mock_perf_counter.return_value = 100.0

        # Track the registered function
        registered_func = None

        def capture_register(func):
            nonlocal registered_func
            registered_func = func

        mock_atexit_register.side_effect = capture_register

        # Call run_carmack (we need to mock the carmack_cli call to avoid running the actual CLI)
        with mock.patch("carmack.__main__.carmack_cli"):
            with mock.patch("carmack.__main__.stderr.print"):
                carmack.__main__.run_carmack()

        # Verify atexit.register was called
        mock_atexit_register.assert_called_once()
        self.assertIsNotNone(registered_func)

        # Simulate time passing and call the registered function
        mock_perf_counter.return_value = 150.5
        with mock.patch("carmack.__main__.log") as mock_log:
            registered_func()

        # Verify format_duration was called with elapsed time (50.5 seconds)
        mock_format_duration.assert_called_once_with(50.5)
        mock_log.info.assert_called_once()

    @mock.patch("carmack.__main__.TagDedup", autospec=True)
    def test_cli_command_bam_tag_deduplicate(self, mock_tag_dedup):
        """Test bam-tag-deduplicate command with all required arguments."""
        # Init
        params = {"output_dir": ".", "prefix": ""}

        # Test
        cmd = (
            ["bam-tag-deduplicate"]
            + [BAM_PATH, BAI_PATH, BC_VALID_PATH]
            + self.assemble_params(params)
        )
        result = self.invoke_cli(cmd)

        # Assert
        self.assertEqual(result.exit_code, 0)
        mock_tag_dedup.assert_called_once_with(BAM_PATH, BAI_PATH, BC_VALID_PATH, umi_map=None)
        mock_tag_dedup.return_value.tag_dedup_reads.assert_called_once_with(False, ".", "")

    @mock.patch("carmack.__main__.TagDedup", autospec=True)
    def test_cli_command_bam_tag_deduplicate_with_dedup_flag(self, mock_tag_dedup):
        """Test bam-tag-deduplicate command with dedup flag enabled."""
        # Init
        params = {"output_dir": ".", "prefix": "test"}

        # Test
        cmd = (
            ["bam-tag-deduplicate"]
            + [BAM_PATH, BAI_PATH, BC_VALID_PATH]
            + self.assemble_params(params)
            + ["--dedup"]
        )
        result = self.invoke_cli(cmd)

        # Assert
        self.assertEqual(result.exit_code, 0)
        mock_tag_dedup.assert_called_once_with(BAM_PATH, BAI_PATH, BC_VALID_PATH, umi_map=None)
        mock_tag_dedup.return_value.tag_dedup_reads.assert_called_once_with(True, ".", "test")

    @mock.patch("carmack.__main__.TagDedup", autospec=True)
    def test_cli_command_bam_tag_deduplicate_with_umi_map(self, mock_tag_dedup):
        """Test bam-tag-deduplicate command threads --umi-map to TagDedup."""
        # Init - any existing file satisfies the click.Path(exists=True) check.
        params = {"output_dir": ".", "prefix": "", "umi-map": BC_VALID_PATH}

        # Test
        cmd = (
            ["bam-tag-deduplicate"]
            + [BAM_PATH, BAI_PATH, BC_VALID_PATH]
            + self.assemble_params(params)
        )
        result = self.invoke_cli(cmd)

        # Assert
        self.assertEqual(result.exit_code, 0)
        mock_tag_dedup.assert_called_once_with(
            BAM_PATH, BAI_PATH, BC_VALID_PATH, umi_map=BC_VALID_PATH
        )

    @mock.patch("carmack.__main__.TagDedup", autospec=True)
    def test_cli_command_bam_tag_deduplicate_default_dedup_false(self, mock_tag_dedup):
        """Test bam-tag-deduplicate command defaults dedup to False."""
        """Test that dedup defaults to False when flag is not provided."""
        cmd = ["bam-tag-deduplicate", BAM_PATH, BAI_PATH, BC_VALID_PATH]
        result = self.invoke_cli(cmd)

        # Assert
        self.assertEqual(result.exit_code, 0)
        mock_tag_dedup.return_value.tag_dedup_reads.assert_called_once()
        # Check that dedup was passed as False (first positional arg)
        call_args = mock_tag_dedup.return_value.tag_dedup_reads.call_args
        self.assertEqual(call_args[0][0], False)  # dedup=False

    @mock.patch("carmack.__main__.TagDedup", autospec=True)
    def test_cli_command_bam_tag_deduplicate_help(self, mock_tag_dedup):
        """Test bam-tag-deduplicate --help displays help message."""
        result = self.invoke_cli(["bam-tag-deduplicate", "--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("bam", result.output.lower())
        self.assertIn("valid_barcodes", result.output.lower())

    @mock.patch("carmack.__main__.BamSplitter", autospec=True)
    def test_cli_command_split_bam(self, mock_bam_splitter):
        """Test split-bam command with all required arguments."""
        # Init
        params = {"output_dir": ".", "prefix": "", "cpu_count": "2"}

        # Test - use tagged BAM for split-bam command
        cmd = ["split-bam"] + [TAGGED_BAM_PATH, BAI_PATH] + self.assemble_params(params)
        result = self.invoke_cli(cmd)

        # Assert
        self.assertEqual(result.exit_code, 0)
        mock_bam_splitter.assert_called_once_with(TAGGED_BAM_PATH, BAI_PATH)
        mock_bam_splitter.return_value.split.assert_called_once_with(".", "", 2)

    @mock.patch("carmack.__main__.BamSplitter", autospec=True)
    def test_cli_command_split_bam_auto_bai(self, mock_bam_splitter):
        """Test split-bam command auto-detects BAI when not provided."""
        # Test without BAI argument - use tagged BAM
        cmd = ["split-bam", TAGGED_BAM_PATH, "--output_dir", "."]
        result = self.invoke_cli(cmd)

        # Should succeed even without explicit BAI
        self.assertEqual(result.exit_code, 0)

    @mock.patch("carmack.__main__.BamSplitter", autospec=True)
    def test_cli_command_split_bam_with_prefix(self, mock_bam_splitter):
        """Test split-bam command with custom prefix."""
        cmd = ["split-bam", TAGGED_BAM_PATH, "--prefix", "split_test", "--cpu_count", "4"]
        result = self.invoke_cli(cmd)

        self.assertEqual(result.exit_code, 0)
        mock_bam_splitter.return_value.split.assert_called_once()

    @mock.patch("carmack.__main__.BamSplitter", autospec=True)
    def test_cli_command_split_bam_help(self, mock_bam_splitter):
        """Test split-bam --help displays help message."""
        result = self.invoke_cli(["split-bam", "--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("split", result.output.lower())

    @mock.patch("carmack.__main__.CellCaller", autospec=True)
    def test_cli_command_call_cells(self, mock_cell_caller):
        """Test call-cells command with all required arguments."""
        # Init - don't include None values in params dict
        params = {
            "output_dir": ".",
            "prefix": "",
            "min_overlap": "1",
        }

        # Test - use tagged BAM for call-cells
        cmd = ["call-cells"] + [BED_PATH, TAGGED_BAM_PATH, BAI_PATH] + self.assemble_params(params)
        result = self.invoke_cli(cmd)

        # Assert
        self.assertEqual(result.exit_code, 0)
        mock_cell_caller.assert_called_once_with(BED_PATH, TAGGED_BAM_PATH, BAI_PATH)
        mock_cell_caller.return_value.compute_matrix.assert_called_once_with(min_overlap=1)
        mock_cell_caller.return_value.export.assert_called_once_with(".", "", None)

    @mock.patch("carmack.__main__.CellCaller", autospec=True)
    def test_cli_command_call_cells_with_visualise(self, mock_cell_caller):
        """Test call-cells command with visualise flag."""
        params = {"output_dir": ".", "prefix": "test", "min_overlap": "1"}

        cmd = (
            ["call-cells"]
            + [BED_PATH, TAGGED_BAM_PATH, BAI_PATH]
            + self.assemble_params(params)
            + ["--visualise"]
        )
        result = self.invoke_cli(cmd)

        self.assertEqual(result.exit_code, 0)
        mock_cell_caller.return_value.make_plot.assert_called_once_with(force_n=None)
        mock_cell_caller.return_value.make_plot.return_value.savefig.assert_called_once()

    @mock.patch("carmack.__main__.CellCaller", autospec=True)
    def test_cli_command_call_cells_with_force_n(self, mock_cell_caller):
        """Test call-cells command with force_n option."""
        params = {"output_dir": ".", "prefix": "", "force_n": "10", "min_overlap": "5"}

        cmd = ["call-cells"] + [BED_PATH, TAGGED_BAM_PATH, BAI_PATH] + self.assemble_params(params)
        result = self.invoke_cli(cmd)

        self.assertEqual(result.exit_code, 0)
        mock_cell_caller.return_value.compute_matrix.assert_called_once_with(min_overlap=5)
        mock_cell_caller.return_value.export.assert_called_once_with(".", "", 10)

    @mock.patch("carmack.__main__.CellCaller", autospec=True)
    def test_cli_command_call_cells_auto_bai(self, mock_cell_caller):
        """Test call-cells command auto-detects BAI when not provided."""
        cmd = ["call-cells", BED_PATH, TAGGED_BAM_PATH, "--output_dir", "."]
        result = self.invoke_cli(cmd)

        self.assertEqual(result.exit_code, 0)

    @mock.patch("carmack.__main__.CellCaller", autospec=True)
    def test_cli_command_call_cells_help(self, mock_cell_caller):
        """Test call-cells --help displays help message."""
        result = self.invoke_cli(["call-cells", "--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("cell", result.output.lower())

    @mock.patch("carmack.__main__.LinearDedup", autospec=True)
    def test_cli_command_linear_dedup(self, mock_linear_dedup):
        """Test linear-dedup command with all required arguments."""
        # Init
        params = {"output_dir": ".", "prefix": ""}

        # Test
        cmd = (
            ["linear-dedup"]
            + [LINEAR_DEDUP_BAM_PATH, LINEAR_DEDUP_BAI_PATH]
            + self.assemble_params(params)
        )
        result = self.invoke_cli(cmd)

        # Assert
        self.assertEqual(result.exit_code, 0)
        mock_linear_dedup.assert_called_once_with(
            LINEAR_DEDUP_BAM_PATH, LINEAR_DEDUP_BAI_PATH, barcode_tag="CB"
        )
        mock_linear_dedup.return_value.linear_dedup_reads.assert_called_once_with(".", "")

    @mock.patch("carmack.__main__.get_bai", autospec=True)
    @mock.patch("carmack.__main__.LinearDedup", autospec=True)
    def test_cli_command_linear_dedup_auto_bai(self, mock_linear_dedup, mock_get_bai):
        """Test linear-dedup command auto-detects BAI when not provided."""
        mock_get_bai.return_value = LINEAR_DEDUP_BAI_PATH

        # Test without BAI argument
        cmd = ["linear-dedup", LINEAR_DEDUP_BAM_PATH, "--output_dir", "."]
        result = self.invoke_cli(cmd)

        # Assert
        self.assertEqual(result.exit_code, 0)
        mock_get_bai.assert_called_once_with(LINEAR_DEDUP_BAM_PATH)
        mock_linear_dedup.assert_called_once_with(
            LINEAR_DEDUP_BAM_PATH, LINEAR_DEDUP_BAI_PATH, barcode_tag="CB"
        )

    @mock.patch("carmack.__main__.LinearDedup", autospec=True)
    def test_cli_command_linear_dedup_default_barcode_tag(self, mock_linear_dedup):
        """Test linear-dedup command defaults barcode_tag to CB when not provided."""
        cmd = ["linear-dedup", LINEAR_DEDUP_BAM_PATH, LINEAR_DEDUP_BAI_PATH, "--output_dir", "."]
        result = self.invoke_cli(cmd)

        # Assert
        self.assertEqual(result.exit_code, 0)
        mock_linear_dedup.assert_called_once_with(
            LINEAR_DEDUP_BAM_PATH, LINEAR_DEDUP_BAI_PATH, barcode_tag="CB"
        )

    @mock.patch("carmack.__main__.LinearDedup", autospec=True)
    def test_cli_command_linear_dedup_custom_barcode_tag(self, mock_linear_dedup):
        """Test linear-dedup command threads --barcode_tag through to LinearDedup."""
        cmd = [
            "linear-dedup",
            LINEAR_DEDUP_BAM_PATH,
            LINEAR_DEDUP_BAI_PATH,
            "--output_dir",
            ".",
            "--barcode_tag",
            "XC",
        ]
        result = self.invoke_cli(cmd)

        # Assert
        self.assertEqual(result.exit_code, 0)
        mock_linear_dedup.assert_called_once_with(
            LINEAR_DEDUP_BAM_PATH, LINEAR_DEDUP_BAI_PATH, barcode_tag="XC"
        )

    @mock.patch("carmack.__main__.LinearDedup", autospec=True)
    def test_cli_command_linear_dedup_custom_barcode_tag_short_flag(self, mock_linear_dedup):
        """Test linear-dedup command threads -b through to LinearDedup."""
        cmd = [
            "linear-dedup",
            LINEAR_DEDUP_BAM_PATH,
            LINEAR_DEDUP_BAI_PATH,
            "-o",
            ".",
            "-b",
            "XC",
        ]
        result = self.invoke_cli(cmd)

        # Assert
        self.assertEqual(result.exit_code, 0)
        mock_linear_dedup.assert_called_once_with(
            LINEAR_DEDUP_BAM_PATH, LINEAR_DEDUP_BAI_PATH, barcode_tag="XC"
        )

    @mock.patch("carmack.__main__.LinearDedup", autospec=True)
    def test_cli_command_linear_dedup_with_prefix(self, mock_linear_dedup):
        """Test linear-dedup command with a custom prefix."""
        cmd = [
            "linear-dedup",
            LINEAR_DEDUP_BAM_PATH,
            LINEAR_DEDUP_BAI_PATH,
            "--prefix",
            "linear_dedup_test",
            "--output_dir",
            ".",
        ]
        result = self.invoke_cli(cmd)

        self.assertEqual(result.exit_code, 0)
        mock_linear_dedup.return_value.linear_dedup_reads.assert_called_once_with(
            ".", "linear_dedup_test"
        )

    @mock.patch("carmack.__main__.LinearDedup", autospec=True)
    def test_cli_command_linear_dedup_default_prefix(self, mock_linear_dedup):
        """Test linear-dedup command defaults prefix to None when not provided."""
        cmd = ["linear-dedup", LINEAR_DEDUP_BAM_PATH, LINEAR_DEDUP_BAI_PATH]
        result = self.invoke_cli(cmd)

        self.assertEqual(result.exit_code, 0)
        mock_linear_dedup.return_value.linear_dedup_reads.assert_called_once_with(".", None)

    @mock.patch("carmack.__main__.LinearDedup", autospec=True)
    def test_cli_command_linear_dedup_help(self, mock_linear_dedup):
        """Test linear-dedup --help displays help message describing the grouping/scoring rule."""
        result = self.invoke_cli(["linear-dedup", "--help"])

        self.assertEqual(result.exit_code, 0)
        output_lower = result.output.lower()
        self.assertIn("bam", output_lower)
        self.assertIn("barcode", output_lower)
        self.assertIn("chromosome", output_lower)
        self.assertIn("strand", output_lower)
        self.assertIn("position", output_lower)
        self.assertIn("score", output_lower)
        # The docstring describes the grouping/scoring rule in carmack's own terms only,
        # never naming any external/downstream pipeline this module might later run inside of.
        self.assertNotIn("cutandrun", output_lower)
        self.assertNotIn("cut&run", output_lower)
        self.assertNotIn("cut&tag", output_lower)
        self.assertNotIn("nf-core", output_lower)
        self.assertNotIn("jacquard", output_lower)

    def test_cli_commands_listed_in_help(self):
        """Test that all commands are listed in the main help output."""
        result = self.invoke_cli(["--help"])

        self.assertEqual(result.exit_code, 0)
        # Check for command names in help output
        self.assertIn("extract-barcodes", result.output)
        self.assertIn("bam-tag-deduplicate", result.output)
        self.assertIn("split-bam", result.output)
        self.assertIn("call-cells", result.output)
        self.assertIn("linear-dedup", result.output)
