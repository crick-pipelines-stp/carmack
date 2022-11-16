import os

from carmack.io.subprocess_stream import SubprocessStream
from ..utils import with_temporary_folder

CONTENT = "test_content"

def test_subprocess_stream_gzip_read(self):
    """Test with read gzip file"""
    stream = SubprocessStream(["gzip", "-c", "-d", "tests/data/test.txt.gz"], mode="r")
    assert stream is not None

    for idx, line in enumerate(stream):
        if idx == 0:
            assert line == "@NB501505:171:H3KMGAFX3:1:21208:17616:17963 1:N:0:AGATCTCGGT"

@with_temporary_folder
def test_subprocess_stream_gzip_write(self, tmp_path):
    """Test with write gzip file"""

    filename = os.path.join(tmp_path, "test.gz")
    file = open(filename, "w")
    stream = SubprocessStream(["gzip", "-c"], stdout=file, mode="w")
    stream.write(CONTENT.encode('UTF-8'))
    stream.close()

    stream = SubprocessStream(["gzip", "-c", "-d", filename], mode="r")
    assert stream is not None

    for idx, line in enumerate(stream):
        print(line)
        if idx == 0:
            assert line.decode('UTF-8') == CONTENT
