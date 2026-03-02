# pylint: disable=missing-function-docstring, missing-class-docstring

import tempfile
import unittest
from unittest import mock

from click.testing import CliRunner

import carmack.__main__


R1_PATH = "tests/data/hydrop_scatac_1_S1_R1_001.fastq.gz"
R2_PATH = "tests/data/hydrop_scatac_1_S1_R3_001.fastq.gz"
CB_PATH = "tests/data/hydrop_scatac_1_S1_R2_001.fastq.gz"
BAM_PATH = "tests/data/hydrop_scatac_1_S1_R1.bam"
BAI_PATH = "tests/data/hydrop_scatac_1_S1_R1.target.sorted.bam.bai"
BC_VALID_PATH = "tests/data/bc_valid.csv"


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

    @mock.patch("carmack.__main__.FastqFilter", autospec=True)
    def test_cli_command_fastq_filter(self, mock_fastq_filter):
        """Test fastq_filter"""

        # Init
        params = {"output_dir": ".", "prefix": ""}

        # Test
        cmd = ["fastq-filter"] + [R1_PATH, R2_PATH, BC_VALID_PATH] + self.assemble_params(params)
        result = self.invoke_cli(cmd)

        # print(mock_fastq_filter.call_args)
        # print(mock_fastq_filter.return_value.filter_valid_reads.call_args)
        # print(result)
        # print(result.output)
        # print(result.exception)

        # Assert
        self.assertTrue(result.exit_code == 0)
        mock_fastq_filter.assert_called_once_with(R1_PATH, R2_PATH)
        # mock_fastq_filter.return_value.filter_valid_reads.assert_called_once_with(
        #     BC_VALID_PATH, params["output_dir"], params["prefix"]
        # )

    @mock.patch("carmack.__main__.BarcodeExtractor", autospec=True)
    def test_cli_command_barcode_extractor(self, mock_barcode_extractor):
        """Test barcode_extractor"""

        # Init
        params = {"chemistry": "hydrop", "output_dir": ".", "prefix": "", "cpu_count": "1"}

        # Test
        cmd = ["extract-barcodes"] + [R1_PATH] + self.assemble_params(params)
        result = self.invoke_cli(cmd)

        print(result.output)
        print(result.exception)

        # Assert
        self.assertTrue(result.exit_code == 0)
        mock_barcode_extractor.assert_called_once_with(R1_PATH, "hydrop", n_workers=1)

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
        with mock.patch("carmack.__main__.carmack_cli") as mock_carmack_cli:
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

    # @mock.patch("carmack.__main__.DuplicateRemoval", autospec=True)
    # def test_cli_command_duplicate_removal(self, mock_duplicate_removal):
    #     """Test duplicate_removal"""

    #     # Init
    #     params = {"output_dir": ".",
    #               "prefix": ''}

    #     # Test
    #     cmd = ["bam-tag-deduplicate"] + [BAM_PATH, BAI_PATH, BC_VALID_PATH] + self.assemble_params(params)
    #     result = self.invoke_cli(cmd)

    #     # print(mock_duplicate_removal.call_args)
    #     # print(mock_duplicate_removal.return_value.tag_and_deduplicate_reads.call_args)
    #     # print(result)
    #     # print(result.output)
    #     # print(result.exception)

    #     # Assert
    #     self.assertTrue(result.exit_code == 0)
    #     mock_duplicate_removal.assert_called_once_with(BAM_PATH, BAI_PATH, BC_VALID_PATH)
    #     mock_duplicate_removal.return_value.tag_and_deduplicate_reads.assert_called_once_with(False, params["output_dir"], False, params["prefix"])
