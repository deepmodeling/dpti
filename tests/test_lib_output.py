import io
import sys
import tempfile
import unittest
from pathlib import Path

from dpti.lib.output import tee_stdout


class TestTeeStdout(unittest.TestCase):
    def test_tee_stdout_writes_to_stdout_and_file(self):
        old_stdout = sys.stdout
        captured = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "result.out"
            try:
                sys.stdout = captured
                with tee_stdout(output):
                    print("hti contribution")
            finally:
                sys.stdout = old_stdout

            self.assertEqual(captured.getvalue(), "hti contribution\n")
            self.assertEqual(output.read_text(), "hti contribution\n")
