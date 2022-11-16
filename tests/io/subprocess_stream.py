from carmack.io.subprocess_stream import SubprocessStream

def test_subprocess_stream_gzip_read(self):
    """Test with read gzip file"""
    stream = SubprocessStream(["gzip", "-c", "-d", "tests/data/test.txt.gz"], mode="r")
    assert stream is not None

    for idx, line in enumerate(stream):
        if idx == 0:
            assert line == "@NB501505:171:H3KMGAFX3:1:21208:17616:17963 1:N:0:AGATCTCGGT"
