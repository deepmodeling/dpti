#!/usr/bin/env python3

import os
import sys
from contextlib import contextmanager


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_stdout(path):
    """Write stdout to both the terminal and a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old_stdout = sys.stdout
    with open(path, "w") as fp:
        sys.stdout = _Tee(old_stdout, fp)
        try:
            yield
        finally:
            sys.stdout = old_stdout
