"""
Helper functions for loading lists of barcodes from files
"""

import os

from carmack.io.gzip_file import GzipFile


def load_barcode_whitelist(path):
    full_path = os.path.abspath(path)
    gz_file = GzipFile(full_path)
    stream = gz_file.open_read_iterator(as_string=True)

    barcodes = []
    for line in stream:
        barcodes.append(line)
    return barcodes
