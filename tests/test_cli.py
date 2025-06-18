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

    @mock.patch("carmack.__main__.BarcodeExtractor", autospec=True)
    def test_cli_command_extract_cell_barcodes(self, mock_barcode_ext):
        """Test extract_cell_barcodes"""

        # Init
        params = {
            "chemistry": "hydrop",
            "max_dist": 2,
            "log_freq": 100,
            "output_dir": ".",
            "prefix": "",
        }

        # Test
        cmd = (
            ["extract-cell-barcodes"] + [R1_PATH, R2_PATH, CB_PATH] + self.assemble_params(params)
        )
        result = self.invoke_cli(cmd)

        # print(mock_barcode_ext.call_args)
        # print(mock_barcode_ext.return_value.extract_cell_barcodes.call_args)
        # print(result)
        # print(result.output)
        # print(result.exception)

        # Assert
        self.assertTrue(result.exit_code == 0)
        mock_barcode_ext.assert_called_once_with(R1_PATH, R2_PATH, CB_PATH, params["chemistry"])
        mock_barcode_ext.return_value.extract_cell_barcodes.assert_called_once_with(
            params["max_dist"], False, params["log_freq"], params["output_dir"], params["prefix"]
        )

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
