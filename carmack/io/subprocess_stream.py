import logging
import os
import subprocess
import sys

from .log_subprocess import LogSubprocess

log = logging.getLogger(__name__)


class SubprocessStream(object):
    """
    Wrap a subprocess that we stream from or stream to. Acts like an open filehandle by
    passing next, write and close down to its pipe.

    Closing the stream checks the child's exit status, so a compressor that dies part-way
    through is reported instead of leaving a truncated file behind for a later stage to
    trip over.
    """

    def __init__(self, *args, **kwargs):
        self.mode = kwargs.pop("mode", "r")
        self.reached_eof = False

        if self.mode == "r":
            kwargs["stdout"] = subprocess.PIPE
        elif self.mode == "w":
            kwargs["stdin"] = subprocess.PIPE
        else:
            raise ValueError("mode %s unsupported" % self.mode)

        kwargs["preexec_fn"] = os.setsid
        sys.stdout.flush()
        sub_proc = LogSubprocess()
        self.proc = sub_proc.Popen(*args, **kwargs)

        if self.mode == "r":
            self.pipe = self.proc.stdout
        elif self.mode == "w":
            self.pipe = self.proc.stdin

    def __enter__(self):
        return self

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.pipe.__next__()
        except StopIteration:
            self.reached_eof = True
            raise

    def write(self, x):
        self.pipe.write(x)

    @property
    def status_is_informative(self) -> bool:
        """
        Whether the child's exit status describes its own work rather than ours.

        A write child is always answerable for how it exited. A read child only becomes
        answerable once the stream has been read to EOF, because closing a read pipe early
        sends SIGPIPE to the child: that status reports on the reader that walked away and
        says nothing about the input. Reads abandoned part-way through are ordinary here -
        a caller takes the first read to validate its header and drops the iterator when
        validation fails - so without this gate every such path would fail spuriously and
        bury the real error underneath.

        Once EOF has been reached the status matters again, and not only for truncation: a
        member with a flipped bit decompresses to the full line count and the full byte
        count with different contents, and gzip reports that solely through its exit
        status.

        Returns:
            True when a non-zero exit status is worth reporting to the caller.
        """
        return self.mode == "w" or self.reached_eof

    def reap(self) -> tuple[int, BrokenPipeError | None]:
        """
        Close the pipe and wait for the child, keeping any error the close reported.

        A BrokenPipeError from the flush is caught rather than allowed to escape, so the
        child whose death caused it is still waited for rather than left orphaned by it.

        Returns:
            A tuple of the child's exit status and the BrokenPipeError raised by closing
            the pipe, or None where the pipe closed cleanly.
        """
        broken = None
        try:
            self.pipe.close()
        except BrokenPipeError as error:
            broken = error
        return self.proc.wait(), broken

    def close(self) -> None:
        """
        Close the stream and report a child that failed at work it is answerable for.

        A BrokenPipeError from the flush is the same event as the child's death and carries
        no root cause of its own - errno 32, no command name, no exit status - so the exit
        status is raised in its place and the flush error is kept as the cause. Where the
        child exited cleanly there is no such story to tell, and the flush error stays the
        best available report.

        Raises:
            subprocess.CalledProcessError: If the child exited non-zero at work it is
                answerable for.
            BrokenPipeError: If the flush failed while the exit status was clean.
        """
        returncode, broken = self.reap()
        if self.status_is_informative and returncode != 0:
            raise subprocess.CalledProcessError(returncode, self.proc.args) from broken
        if broken is not None:
            raise broken

    def __exit__(self, tp, val, tb) -> None:
        """
        Close the stream without burying an exception that is already in flight.

        A BrokenPipeError from the body is the child's death seen from this end, so
        replacing it with the exit status while keeping it as the cause strictly informs.
        Any other exception, GeneratorExit included, is a real fault of its own that must
        never disappear behind a message about a compressor, so the status is logged and
        the original exception is left to propagate untouched.
        """
        if tp is None:
            self.close()
            return

        returncode, _ = self.reap()
        if self.status_is_informative and returncode != 0:
            if issubclass(tp, BrokenPipeError):
                raise subprocess.CalledProcessError(returncode, self.proc.args) from val
            log.warning(
                f"Command {self.proc.args} exited with status {returncode} while another "
                "error was already being raised."
            )
