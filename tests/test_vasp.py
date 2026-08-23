import os
import tempfile
import unittest

from dpti.lib.vasp import regulate_poscar, sort_poscar

POSCAR = """water
1.0
1 0 0
0 1 0
0 0 1
O H O
1 2 1
Direct
0.0 0.0 0.0
0.1 0.1 0.1
0.2 0.2 0.2
0.3 0.3 0.3
"""


class TestPoscarGrouping(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.input_path = os.path.join(self.tempdir.name, "POSCAR")
        with open(self.input_path, "w") as fp:
            fp.write(POSCAR)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_regulate_retains_unlabeled_coordinates(self):
        output_path = os.path.join(self.tempdir.name, "regulated.POSCAR")

        regulate_poscar(self.input_path, output_path)

        with open(output_path) as fp:
            lines = fp.read().splitlines()
        self.assertEqual(lines[5], "O H")
        self.assertEqual(lines[6], "2 2")
        self.assertEqual(
            lines[8:12],
            ["0.0 0.0 0.0", "0.3 0.3 0.3", "0.1 0.1 0.1", "0.2 0.2 0.2"],
        )

    def test_sort_uses_header_element_ownership(self):
        output_path = os.path.join(self.tempdir.name, "sorted.POSCAR")

        sort_poscar(self.input_path, output_path, ["H", "O"])

        with open(output_path) as fp:
            lines = fp.read().splitlines()
        self.assertEqual(lines[5], "H O")
        self.assertEqual(lines[6], "2 2")
        self.assertEqual(
            lines[8:12],
            ["0.1 0.1 0.1", "0.2 0.2 0.2", "0.0 0.0 0.0", "0.3 0.3 0.3"],
        )


if __name__ == "__main__":
    unittest.main()
