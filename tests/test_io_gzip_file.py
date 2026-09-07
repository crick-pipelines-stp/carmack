# pylint: disable=missing-function-docstring, missing-class-docstring

import gzip
import os
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from carmack.io.gzip_file import GzipFile, resolve_gzip_write_command
from tests.utils import (
    gzip_bytes,
    read_gzip_text,
    with_temporary_folder,
    write_executable_stub,
)

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

SUBPROCESS_STREAM_MODULE = "carmack.io.subprocess_stream"

# Exit status the stub compressor reports, chosen so that seeing it come back out of the
# write stream proves it travelled from the child rather than being invented.
STUB_EXIT_STATUS = 3

# A compressor that drains its input and then fails writing its own output - the shape a
# full filesystem produces - leaves only this behind in place of the whole payload.
STUB_PARTIAL_OUTPUT = "PARTIAL"

# The corruption fixtures share one payload. Its decompressed size is far past the 64 KiB
# pipe buffer, so a reader that abandons the stream always leaves gzip blocked on a write
# and gzip always dies of SIGPIPE. A smaller payload lets gzip finish and exit zero before
# the pipe closes, which would make the abandonment test pass for the wrong reason. The
# fixed-width line format matters too: with this payload a single flipped bit mid-member
# still decompresses to exactly the right number of lines and exactly the right number of
# bytes, so nothing about the data betrays the damage.
CORRUPTION_LINE_COUNT = 10000
CORRUPTION_TEXT = "".join(f"line {index:08d}\n" for index in range(CORRUPTION_LINE_COUNT))
CORRUPTION_BYTES = CORRUPTION_TEXT.encode("UTF-8")

# Appended when building a file that holds one complete member followed by bytes that are
# not gzip data at all.
TRAILING_GARBAGE = b"this is not gzip data\n"


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


class TestGzipFileWriteFailure(unittest.TestCase):
    def setUp(self):
        resolve_gzip_write_command.cache_clear()
        self.addCleanup(resolve_gzip_write_command.cache_clear)

    @with_temporary_folder
    def test_gzip_file_write_stream_reports_a_compressor_that_died(self, tmp_path):
        """Test a stub compressor found through PATH is reported when it exits non-zero"""
        stub_folder = os.path.join(tmp_path, "bin")
        os.makedirs(stub_folder)

        # The resolver returns the bare name "pigz", so exec resolves it through PATH at
        # the moment the child starts. A stub of that name earlier on PATH therefore
        # substitutes for the real compressor, and the resolver, the PATH lookup and the
        # exit status all take part rather than being mocked out.
        write_executable_stub(
            stub_folder,
            "pigz",
            f"cat > /dev/null; printf '{STUB_PARTIAL_OUTPUT}'; exit {STUB_EXIT_STATUS}",
        )
        filename = os.path.join(tmp_path, "output.gz")
        stub_path = f"{stub_folder}{os.pathsep}{os.environ['PATH']}"

        with patch.dict(os.environ, {"PATH": stub_path}):
            gz_file = GzipFile(filename)
            self.assertEqual(gz_file.write_command[0], "pigz")

            wstream = gz_file.open_write_stream()
            for line in ROUND_TRIP_LINES:
                GzipFile.write_string(wstream, f"{line}\n")

            with self.assertRaises(subprocess.CalledProcessError) as caught:
                wstream.close()

        self.assertEqual(caught.exception.returncode, STUB_EXIT_STATUS)

        # Every write succeeded and the whole payload still went missing: what is left on
        # disk is the stub's own few bytes, and no gzip reader can make anything of it.
        self.assertEqual(os.path.getsize(filename), len(STUB_PARTIAL_OUTPUT))
        with self.assertRaises(gzip.BadGzipFile):
            read_gzip_text(filename)


class TestGzipFileReadFailure(unittest.TestCase):
    def write_corruption_fixture(self, folder, name, content):
        """
        Write one gzip fixture into a folder and return its path.

        Args:
            folder: Directory to write the fixture into.
            name: File name for the fixture.
            content: Exact bytes to write, valid gzip data or otherwise.

        Returns:
            The path of the fixture, as a str.
        """

        path = Path(folder) / name
        path.write_bytes(content)
        return str(path)

    def read_all(self, filename):
        """
        Read a gzip file through GzipFile to EOF, returning every line it yielded.

        Args:
            filename: Path of the file to read.

        Returns:
            The lines yielded, as a list of bytes objects.
        """

        return list(GzipFile(filename).open_read_iterator())

    @with_temporary_folder
    def test_gzip_file_read_reports_a_corrupted_member_that_looked_complete(self, tmp_path):
        """Test a single flipped bit is reported even though every line still arrived"""
        healthy = gzip_bytes(CORRUPTION_TEXT)
        corrupted = bytearray(healthy)
        corrupted[len(healthy) // 2] ^= 0x01
        filename = self.write_corruption_fixture(tmp_path, "corrupted.gz", bytes(corrupted))

        lines = []
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            for line in GzipFile(filename).open_read_iterator():
                lines.append(line)

        self.assertEqual(caught.exception.returncode, 1)

        # This is the case the read-side check exists for. The reader saw the full line
        # count and the full byte count, so nothing in the data marks it as damaged, yet
        # the contents are not what was compressed. Without the exit status a caller has
        # no way at all to tell this apart from a healthy file.
        body = b"".join(lines)
        self.assertEqual(len(lines), CORRUPTION_LINE_COUNT)
        self.assertEqual(len(body), len(CORRUPTION_BYTES))
        self.assertNotEqual(body, CORRUPTION_BYTES)

    @with_temporary_folder
    def test_gzip_file_read_reports_garbage_after_a_complete_member(self, tmp_path):
        """Test bytes that are not gzip data following a complete member are reported"""
        filename = self.write_corruption_fixture(
            tmp_path, "trailing_garbage.gz", gzip_bytes(CORRUPTION_TEXT) + TRAILING_GARBAGE
        )

        lines = []
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            for line in GzipFile(filename).open_read_iterator():
                lines.append(line)

        # Pinned deliberately as a behaviour change. Reading is not parallelised, so the
        # read path always runs gzip, and gzip calls trailing junk a warning worth an exit
        # status of 2 where pigz exits zero and says nothing. A file shaped like this has
        # had something appended to it that the writer never intended, so reporting it is
        # the wanted outcome even though the member itself decompressed cleanly.
        self.assertEqual(caught.exception.returncode, 2)
        self.assertEqual(len(lines), CORRUPTION_LINE_COUNT)

    @with_temporary_folder
    def test_gzip_file_read_accepts_concatenated_members(self, tmp_path):
        """Test a multi-member file reads every member through without being reported"""
        healthy = gzip_bytes(CORRUPTION_TEXT)
        filename = self.write_corruption_fixture(tmp_path, "two_members.gz", healthy + healthy)

        lines = self.read_all(filename)

        # The false-positive guard for the trailing-garbage decision above. Concatenated
        # members are ordinary in sequencing data, so the check must not mistake a second
        # member for junk: both members are read and nothing is raised.
        self.assertEqual(len(lines), CORRUPTION_LINE_COUNT * 2)
        self.assertEqual(b"".join(lines), CORRUPTION_BYTES * 2)

    @with_temporary_folder
    def test_gzip_file_read_reports_an_empty_file(self, tmp_path):
        """Test a zero-byte file is reported rather than read as an empty stream"""
        filename = self.write_corruption_fixture(tmp_path, "empty.gz", b"")

        with self.assertRaises(subprocess.CalledProcessError) as caught:
            self.read_all(filename)

        self.assertEqual(caught.exception.returncode, 1)

    @with_temporary_folder
    def test_gzip_file_read_iterator_abandoned_early_reports_nothing(self, tmp_path):
        """Test closing a read iterator part-way through neither raises nor warns"""
        filename = self.write_corruption_fixture(
            tmp_path, "healthy.gz", gzip_bytes(CORRUPTION_TEXT)
        )

        # This is the production shape: a caller takes the first read to validate its
        # header, and drops the iterator where validation fails. gzip is left blocked on a
        # write and dies of SIGPIPE, which says nothing about the file, so an abandoned
        # iterator must close in silence rather than raising or logging over the real error.
        with self.assertNoLogs(SUBPROCESS_STREAM_MODULE, level="WARNING"):
            reads = GzipFile(filename).open_read_iterator()
            first_line = next(reads, None)
            reads.close()

        self.assertIsNotNone(first_line)
