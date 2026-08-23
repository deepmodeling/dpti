import unittest

import numpy as np

from dpti.lib.dump import get_dumpbox


class TestDumpBox(unittest.TestCase):
    def test_orthogonal_box_defaults_to_zero_tilt(self):
        """Standard two-column BOX BOUNDS lines describe an orthogonal box."""
        lines = [
            "ITEM: BOX BOUNDS pp pp pp",
            "0 10",
            "-1 9",
            "2 12",
            "ITEM: ATOMS id type x y z",
        ]

        bounds, tilt = get_dumpbox(lines)

        np.testing.assert_allclose(bounds, [[0, 10], [-1, 9], [2, 12]])
        np.testing.assert_allclose(tilt, [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
