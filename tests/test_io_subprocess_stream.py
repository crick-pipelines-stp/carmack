# pylint: disable=missing-function-docstring, missing-class-docstring


import os
import signal
import subprocess
import unittest
from pathlib import Path

from carmack.io.subprocess_stream import SubprocessStream
from tests.utils import gzip_bytes, with_temporary_folder, write_executable_stub

CONTENT = "test_content"
READ_NAME = "@NB501505:171:H3KMGAFX3:1:21208:17616:17963 1:N:0:AGATCTCGGT"

SUBPROCESS_STREAM_MODULE = "carmack.io.subprocess_stream"

# Exit status the stub compressors report. Nothing else in the pipeline uses it, so seeing
# it come back out of close() proves it travelled from the child rather than being invented.
STUB_EXIT_STATUS = 3

# A stub that drains stdin and then fails writing its own output leaves this in place of the
# payload, which is the shape an out-of-space compressor leaves behind.
STUB_PARTIAL_OUTPUT = "PARTIAL"

# Larger than the 64 KiB pipe buffer by a wide margin, so a stub that stops reading is
# certain to have exited while the parent is still writing. Nothing here waits on timing.
UNBUFFERABLE_PAYLOAD = b"x" * (4 * 1024 * 1024)

# Smaller than the 8 KiB buffer Python puts in front of the pipe, so this payload is still
# sitting in that buffer when write() returns and only reaches the pipe on the flush inside
# close(). That is what makes the flush, rather than the write, the first thing to fail.
BUFFERED_PAYLOAD = b"x" * 4096

# Decompressed output far past the 64 KiB pipe buffer, so a reader that stops early always
# leaves gzip blocked on a write and gzip always dies of SIGPIPE. A smaller payload lets
# gzip finish and exit zero before the pipe closes, which would make the early-close tests
# pass for the wrong reason.
LARGE_READ_LINE_COUNT = 10000
LARGE_READ_TEXT = "".join(f"line {index:08d}\n" for index in range(LARGE_READ_LINE_COUNT))

# Small enough that the whole decompressed prefix of the truncated member fits in the pipe
# buffer. gzip therefore always reaches its own non-zero exit before the reader closes the
# pipe, so the status seen on an early close is the input error rather than SIGPIPE.
SMALL_READ_LINE_COUNT = 200
SMALL_READ_TEXT = "".join(f"line {index}\n" for index in range(SMALL_READ_LINE_COUNT))

# How far into a read stream the early-close tests get before abandoning it.
EARLY_CLOSE_LINES = 5

# Fraction of a gzip member kept when building a truncated one.
TRUNCATION_FRACTION = 0.5

FAILURE_MESSAGE = "the caller itself is broken"


def read_lines(stream, limit=None):
    """
    Consume a stream, returning the lines read and stopping early at an optional limit.

    Args:
        stream: Stream to iterate, yielding one bytes object per line.
        limit: Number of lines to read before stopping, or None to read to EOF.

    Returns:
        The lines read, as a list of bytes objects.
    """

    lines = []
    for line in stream:
        lines.append(line)
        if limit is not None and len(lines) == limit:
            break
    return lines


class TestSubprocessStream(unittest.TestCase):
    def test_subprocess_stream_gzip_read(self):
        """Test with read gzip file"""
        with SubprocessStream(
            ["gzip", "-c", "-d", "tests/data/sc_10k.fastq.gz"], mode="r"
        ) as stream:
            assert stream is not None

            for idx, line in enumerate(stream):
                if idx == 0:
                    line_str = line.decode("UTF-8").strip()
                    self.assertEqual(line_str, READ_NAME)

    @with_temporary_folder
    def test_subprocess_stream_gzip_write(self, tmp_path):
        """Test with write gzip file"""
        filename = os.path.join(tmp_path, "test.gz")
        file = open(filename, "w")
        stream = SubprocessStream(["gzip", "-c"], stdout=file, mode="w")
        stream.write(CONTENT.encode("UTF-8"))
        stream.close()

        stream = SubprocessStream(["gzip", "-c", "-d", filename], mode="r")
        assert stream is not None

        for idx, line in enumerate(stream):
            if idx == 0:
                self.assertEqual(line.decode("UTF-8"), CONTENT)


class TestSubprocessStreamWriteFailure(unittest.TestCase):
    """Tests that a write-mode child's exit status is not discarded"""

    @with_temporary_folder
    def test_close_reports_compressor_that_died_after_draining_stdin(self, tmp_path):
        """Test close reports a compressor that accepted every byte and then failed"""
        stub = write_executable_stub(
            tmp_path,
            "draining-compressor",
            f"cat > /dev/null; printf '{STUB_PARTIAL_OUTPUT}'; exit {STUB_EXIT_STATUS}",
        )
        filename = os.path.join(tmp_path, "output.gz")

        with open(filename, "w") as handle:
            stream = SubprocessStream([stub], stdout=handle, mode="w")
            stream.write(UNBUFFERABLE_PAYLOAD)

            with self.assertRaises(subprocess.CalledProcessError) as caught:
                stream.close()

        self.assertEqual(caught.exception.returncode, STUB_EXIT_STATUS)
        self.assertIn(stub, str(caught.exception))

        # The payload really did vanish: the stub read all of it and wrote only its own
        # few bytes, which is the failure the discarded exit status used to hide.
        self.assertEqual(os.path.getsize(filename), len(STUB_PARTIAL_OUTPUT))

    @with_temporary_folder
    def test_close_reports_exit_status_with_the_flush_failure_as_its_cause(self, tmp_path):
        """Test a failed flush is reported as the exit status that explains it"""
        stub = write_executable_stub(tmp_path, "dead-compressor", f"exit {STUB_EXIT_STATUS}")
        filename = os.path.join(tmp_path, "output.gz")

        with open(filename, "w") as handle:
            stream = SubprocessStream([stub], stdout=handle, mode="w")

            # Reaping here leaves the pipe with no reader, so the buffered write below
            # cannot reach it and close() has to flush into a broken pipe. Waiting again
            # inside close() must still yield the status rather than losing it.
            stream.proc.wait()
            stream.write(BUFFERED_PAYLOAD)

            with self.assertRaises(subprocess.CalledProcessError) as caught:
                stream.close()

        self.assertEqual(caught.exception.returncode, STUB_EXIT_STATUS)
        self.assertIsInstance(caught.exception.__cause__, BrokenPipeError)

    @with_temporary_folder
    def test_close_raises_the_flush_failure_when_the_compressor_succeeded(self, tmp_path):
        """Test a failed flush still surfaces when the exit status has no story to tell"""
        stub = write_executable_stub(tmp_path, "quitting-compressor", "exit 0")
        filename = os.path.join(tmp_path, "output.gz")

        with open(filename, "w") as handle:
            stream = SubprocessStream([stub], stdout=handle, mode="w")
            stream.proc.wait()
            stream.write(BUFFERED_PAYLOAD)

            with self.assertRaises(BrokenPipeError):
                stream.close()

    @with_temporary_folder
    def test_exit_reports_exit_status_when_the_write_broke_the_pipe(self, tmp_path):
        """Test a broken pipe during a write is replaced by the status behind it"""
        stub = write_executable_stub(
            tmp_path,
            "deaf-compressor",
            f"head -c 2000 > /dev/null; printf x; exit {STUB_EXIT_STATUS}",
        )
        filename = os.path.join(tmp_path, "output.gz")

        with open(filename, "w") as handle:
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                with SubprocessStream([stub], stdout=handle, mode="w") as stream:
                    stream.write(UNBUFFERABLE_PAYLOAD)

        self.assertEqual(caught.exception.returncode, STUB_EXIT_STATUS)
        self.assertIsInstance(caught.exception.__cause__, BrokenPipeError)

    @with_temporary_folder
    def test_exit_warns_and_leaves_an_unrelated_exception_untouched(self, tmp_path):
        """Test an exception from the caller propagates unchanged, with the status logged"""
        stub = write_executable_stub(tmp_path, "dead-compressor", f"exit {STUB_EXIT_STATUS}")
        filename = os.path.join(tmp_path, "output.gz")

        with open(filename, "w") as handle:
            with self.assertLogs(SUBPROCESS_STREAM_MODULE, level="WARNING") as logged:
                with self.assertRaises(ValueError) as caught:
                    with SubprocessStream([stub], stdout=handle, mode="w"):
                        raise ValueError(FAILURE_MESSAGE)

        self.assertEqual(str(caught.exception), FAILURE_MESSAGE)

        message = " ".join(record.getMessage() for record in logged.records)
        self.assertIn(stub, message)

        # The stub path can contain digits of its own, so the status is looked for in what
        # is left of the message once the command has been taken out of it.
        self.assertIn(str(STUB_EXIT_STATUS), message.replace(stub, ""))


class TestSubprocessStreamReadFailure(unittest.TestCase):
    """Tests that a read-mode child's exit status is judged against how far reading got"""

    @with_temporary_folder
    def test_read_to_eof_reports_nothing_for_a_healthy_member(self, tmp_path):
        """Test a gzip file read all the way through closes without complaint"""
        filename = os.path.join(tmp_path, "healthy.gz")
        Path(filename).write_bytes(gzip_bytes(LARGE_READ_TEXT))

        with SubprocessStream(["gzip", "-c", "-d", filename], mode="r") as stream:
            lines = read_lines(stream)

        self.assertEqual(len(lines), LARGE_READ_LINE_COUNT)

    @with_temporary_folder
    def test_close_before_eof_ignores_the_signal_it_caused(self, tmp_path):
        """Test abandoning a healthy read stream is not reported as an input failure"""
        filename = os.path.join(tmp_path, "healthy.gz")
        Path(filename).write_bytes(gzip_bytes(LARGE_READ_TEXT))

        stream = SubprocessStream(["gzip", "-c", "-d", filename], mode="r")
        lines = read_lines(stream, limit=EARLY_CLOSE_LINES)
        stream.close()

        self.assertEqual(len(lines), EARLY_CLOSE_LINES)

        # gzip was still writing when the pipe went away, so it died of SIGPIPE. That
        # status is a report on the reader, not on the file, and must not be raised.
        self.assertEqual(stream.proc.returncode, -signal.SIGPIPE)

    @with_temporary_folder
    def test_close_before_eof_ignores_a_real_input_error(self, tmp_path):
        """Test abandoning a truncated read stream is not reported either"""
        filename = os.path.join(tmp_path, "truncated.gz")
        healthy = gzip_bytes(SMALL_READ_TEXT)
        Path(filename).write_bytes(healthy[: int(len(healthy) * TRUNCATION_FRACTION)])

        stream = SubprocessStream(["gzip", "-c", "-d", filename], mode="r")
        lines = read_lines(stream, limit=EARLY_CLOSE_LINES)
        stream.close()

        self.assertEqual(len(lines), EARLY_CLOSE_LINES)

        # The whole truncated prefix fitted in the pipe buffer, so gzip reported the
        # damage before the pipe closed. The rule is still gated on having reached EOF:
        # a reader that stopped early has no idea whether the rest was ever going to
        # arrive, so this status stays unreported rather than being raised at a caller
        # who deliberately read only part of the file.
        self.assertEqual(stream.proc.returncode, 1)
