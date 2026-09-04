# pylint: disable=missing-function-docstring, missing-class-docstring

import os
import shutil
import unittest
from unittest.mock import patch

from carmack.io.gzip_file import GzipFile, resolve_gzip_write_command
from tests.utils import read_gzip_text, with_temporary_folder

TEST_NAME = "@NB501505:171:H3KMGAFX3:1:21208:17616:17963 1:N:0:AGATCTCGGT"

# The production module uses from-imports, so the patch targets live on the module under
# test rather than on shutil/os.
GZIP_FILE_MODULE = "carmack.io.gzip_file"

# pigz is not guaranteed on every host, so no test may assume it is installed.
PIGZ_PATH = shutil.which("pigz")

# pigz splits its input into 128 KiB blocks and compresses each on a different thread, so a
# payload has to clear several block boundaries before a round trip exercises the parallel
# path at all. 8000 records is roughly 570 KiB, or about four blocks over the four threads.
ROUND_TRIP_LINES = tuple(f"{TEST_NAME} record {index}" for index in range(8000))
ROUND_TRIP_TEXT = "".join(f"{line}\n" for line in ROUND_TRIP_LINES)


class TestGzipFile(unittest.TestCase):
    def test_gzip_file_read(self):
        """Test reading gzip fastq file"""
        gz_file = GzipFile("tests/data/sc_10k.fastq.gz")

        for line in gz_file.open_read_iterator(as_string=True):
            self.assertEqual(line, TEST_NAME)
            break

    def test_gzip_file_read_raw(self):
        """Test reading gzip file thats actually not compressed"""
        raw_file = GzipFile("tests/data/small.fastq")

        for line in raw_file.open_read_iterator(as_string=True):
            self.assertEqual(line, TEST_NAME)
            break

    @with_temporary_folder
    def test_gzip_file_write(self, tmp_path):
        """Test with write gzip file"""
        filename = os.path.join(tmp_path, "test.gz")
        gz_file = GzipFile(filename)
        wstream = gz_file.open_write_stream()
        GzipFile.write_string(wstream, TEST_NAME)
        wstream.close()

        gz_file = GzipFile(filename)

        for line in gz_file.open_read_iterator(as_string=True):
            self.assertEqual(line, TEST_NAME)
            break


class TestResolveGzipWriteCommand(unittest.TestCase):
    def setUp(self):
        # GzipFile.__init__ populates the cache, so without a clear the result of one test
        # leaks into the next and the outcome depends on test ordering. Clearing on the way
        # out as well stops a mocked result escaping into other test modules, which would
        # otherwise write their gzip output with whichever compressor was last patched in.
        resolve_gzip_write_command.cache_clear()
        self.addCleanup(resolve_gzip_write_command.cache_clear)

    def test_resolve_gzip_write_command_returns_pigz_when_available(self):
        """Test the resolver picks pigz with the configured thread count when it is on PATH"""
        with (
            patch(f"{GZIP_FILE_MODULE}.which", return_value="/usr/bin/pigz"),
            patch(f"{GZIP_FILE_MODULE}.cpu_count", return_value=8),
        ):
            command = resolve_gzip_write_command()

        self.assertEqual(command, ("pigz", "-c", "-p", "4"))

    def test_resolve_gzip_write_command_falls_back_to_gzip_when_pigz_missing(self):
        """Test the resolver falls back to gzip when pigz is not on PATH"""
        with patch(f"{GZIP_FILE_MODULE}.which", return_value=None):
            command = resolve_gzip_write_command()

        self.assertEqual(command, ("gzip", "-c"))

    def test_resolve_gzip_write_command_warns_when_pigz_missing(self):
        """Test the fallback to gzip is announced by a warning naming both commands"""
        with patch(f"{GZIP_FILE_MODULE}.which", return_value=None):
            with self.assertLogs(GZIP_FILE_MODULE, level="WARNING") as logged:
                resolve_gzip_write_command()

        # The formatted output embeds the logger name, which contains "gzip", so the
        # assertion has to run against the message alone.
        message = logged.records[0].getMessage()
        self.assertIn("pigz", message)
        self.assertIn("gzip", message)

    def test_resolve_gzip_write_command_clamps_threads_to_cpu_count(self):
        """Test the thread count never exceeds the number of available CPUs"""
        with (
            patch(f"{GZIP_FILE_MODULE}.which", return_value="/usr/bin/pigz"),
            patch(f"{GZIP_FILE_MODULE}.cpu_count", return_value=2),
        ):
            command = resolve_gzip_write_command()

        self.assertEqual(command, ("pigz", "-c", "-p", "2"))

    def test_resolve_gzip_write_command_handles_unknown_cpu_count(self):
        """Test an unknown CPU count falls back to a single thread rather than failing"""
        with (
            patch(f"{GZIP_FILE_MODULE}.which", return_value="/usr/bin/pigz"),
            patch(f"{GZIP_FILE_MODULE}.cpu_count", return_value=None),
        ):
            command = resolve_gzip_write_command()

        self.assertEqual(command, ("pigz", "-c", "-p", "1"))

    def test_resolve_gzip_write_command_is_memoised(self):
        """Test repeated calls reuse the first result instead of searching PATH again"""
        with (
            patch(f"{GZIP_FILE_MODULE}.which", return_value="/usr/bin/pigz") as which_mock,
            patch(f"{GZIP_FILE_MODULE}.cpu_count", return_value=8),
        ):
            first = resolve_gzip_write_command()
            second = resolve_gzip_write_command()

        self.assertEqual(which_mock.call_count, 1)
        self.assertEqual(first, second)


class TestGzipFileWriteCommand(unittest.TestCase):
    def setUp(self):
        resolve_gzip_write_command.cache_clear()
        self.addCleanup(resolve_gzip_write_command.cache_clear)

    def test_gzip_file_write_command_for_gzip_suffix(self):
        """Test a gzip file takes its write command from the resolver"""
        with (
            patch(f"{GZIP_FILE_MODULE}.which", return_value="/usr/bin/pigz"),
            patch(f"{GZIP_FILE_MODULE}.cpu_count", return_value=8),
        ):
            gz_file = GzipFile("x.gz")

            self.assertEqual(gz_file.write_command, resolve_gzip_write_command())

        self.assertEqual(gz_file.write_command, ("pigz", "-c", "-p", "4"))

    def test_gzip_file_write_command_for_lz4_suffix_ignores_pigz(self):
        """Test an lz4 file keeps its own write command even when pigz is available"""
        with (
            patch(f"{GZIP_FILE_MODULE}.which", return_value="/usr/bin/pigz"),
            patch(f"{GZIP_FILE_MODULE}.cpu_count", return_value=8),
        ):
            lz4_file = GzipFile("x.lz4")

        self.assertEqual(lz4_file.compressor, "lz4")
        self.assertEqual(lz4_file.write_command, ("lz4", "-c"))

    def test_gzip_file_write_command_none_for_unknown_suffix(self):
        """Test an uncompressed file has neither a compressor nor a write command"""
        plain_file = GzipFile("x.fastq")

        self.assertIsNone(plain_file.compressor)
        self.assertIsNone(plain_file.write_command)

    def test_gzip_file_compressor_unchanged_for_gzip_suffix(self):
        """Test a gzip file still reads through gzip even when pigz is available"""
        with (
            patch(f"{GZIP_FILE_MODULE}.which", return_value="/usr/bin/pigz"),
            patch(f"{GZIP_FILE_MODULE}.cpu_count", return_value=8),
        ):
            gz_file = GzipFile("x.gz")

        # Pinned deliberately: the change is narrowed to the write path, so the read path
        # keeps using gzip. A future edit that swaps this to pigz is a scope regression.
        self.assertEqual(gz_file.compressor, "gzip")


class TestGzipFileWriteRoundTrip(unittest.TestCase):
    def setUp(self):
        resolve_gzip_write_command.cache_clear()
        self.addCleanup(resolve_gzip_write_command.cache_clear)

    def write_payload(self, filename):
        """Write the shared payload through the real write stream"""
        gz_file = GzipFile(filename)
        wstream = gz_file.open_write_stream()

        for line in ROUND_TRIP_LINES:
            GzipFile.write_string(wstream, f"{line}\n")

        wstream.close()

    @with_temporary_folder
    def test_gzip_file_write_stream_round_trips_through_stdlib_gzip(self, tmp_path):
        """Test output written by the write stream is readable by the stdlib gzip module"""
        filename = os.path.join(tmp_path, "round_trip.gz")

        self.write_payload(filename)

        self.assertEqual(read_gzip_text(filename), ROUND_TRIP_TEXT)

    @unittest.skipUnless(PIGZ_PATH is not None, "pigz is not installed on this host")
    @with_temporary_folder
    def test_gzip_file_write_stream_output_identical_for_pigz_and_gzip(self, tmp_path):
        """Test pigz and gzip output decompress to identical contents"""
        pigz_filename = os.path.join(tmp_path, "written_by_pigz.gz")
        gzip_filename = os.path.join(tmp_path, "written_by_gzip.gz")

        with patch(f"{GZIP_FILE_MODULE}.which", return_value=PIGZ_PATH):
            self.write_payload(pigz_filename)

        # The first write filled the cache, so the fallback branch needs a clear.
        resolve_gzip_write_command.cache_clear()

        with patch(f"{GZIP_FILE_MODULE}.which", return_value=None):
            self.write_payload(gzip_filename)

        # The compressed bytes legitimately differ between the two compressors, so only the
        # decompressed contents are compared.
        pigz_text = read_gzip_text(pigz_filename)
        gzip_text = read_gzip_text(gzip_filename)

        self.assertEqual(pigz_text, gzip_text)
        self.assertEqual(pigz_text, ROUND_TRIP_TEXT)
