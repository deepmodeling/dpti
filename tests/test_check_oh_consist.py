import importlib.util
import os
import subprocess
import sys
import unittest


class TestCheckOhConsist(unittest.TestCase):
    def setUp(self):
        self.repository_root = os.path.abspath("..")
        self.script = os.path.join(
            self.repository_root, "tools", "check_oh_consist.py"
        )

    def test_import_has_no_dump_file_side_effect(self):
        """Loading the tool defines its API without reading dump.hti."""
        spec = importlib.util.spec_from_file_location("check_oh_consist", self.script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertTrue(callable(module.get_oh_distance_stats))

    def test_help_runs_from_repository_root(self):
        result = subprocess.run(
            [sys.executable, self.script, "--help"],
            cwd=self.repository_root,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("LAMMPS dump trajectory", result.stdout)


if __name__ == "__main__":
    unittest.main()
