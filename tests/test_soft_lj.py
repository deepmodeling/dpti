import tempfile
import unittest
from pathlib import Path

import numpy as np

from dpti.soft_lj import (
    global_rmse,
    load_deepmd,
    pair_index,
    pair_labels,
    pair_lists,
    predict_one_frame,
)


class TestSoftLJ(unittest.TestCase):
    def test_pair_indices_for_three_types(self):
        actual = pair_index(np.array([0, 0, 1, 2, 2]), np.array([0, 2, 1, 0, 2]), 3)
        np.testing.assert_array_equal(actual, [0, 2, 3, 2, 5])
        self.assertEqual(
            pair_labels(["A", "B", "C"]),
            ["A-A", "A-B", "A-C", "B-B", "B-C", "C-C"],
        )

    def test_energy_force_are_consistent(self):
        coord = np.array([[0.0, 0.0, 0.0], [1.7, 0.2, 0.0], [0.3, 2.1, 0.1]])
        box = np.eye(3) * 10.0
        atype = np.array([0, 1, 0])
        ii, jj, pp = pair_lists(atype, 2)
        epsilon = np.array([0.2, 0.3, 0.4])
        sigma = np.array([1.1, 1.2, 1.3])
        activation = np.array([0.5, 0.6, 0.7])
        _, force = predict_one_frame(
            coord, box, ii, jj, pp, epsilon, sigma, activation, 1.0, 0.5, 6.0
        )
        step = 1e-6
        shifted_plus = coord.copy()
        shifted_minus = coord.copy()
        shifted_plus[1, 0] += step
        shifted_minus[1, 0] -= step
        energy_plus, _ = predict_one_frame(
            shifted_plus, box, ii, jj, pp, epsilon, sigma, activation, 1.0, 0.5, 6.0
        )
        energy_minus, _ = predict_one_frame(
            shifted_minus, box, ii, jj, pp, epsilon, sigma, activation, 1.0, 0.5, 6.0
        )
        self.assertAlmostEqual(
            force[1, 0], -(energy_plus - energy_minus) / (2 * step), places=6
        )
        np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1e-12)

    def test_global_rmse_uses_one_energy_offset(self):
        metrics = global_rmse(
            np.array([0.0, 2.0, 10.0, 14.0]),
            np.array([0.0, 1.0, 10.0, 11.0]),
            np.zeros((4, 1, 3)),
            np.ones((4, 1, 3)),
            1,
        )
        self.assertAlmostEqual(metrics["e_rmse_atom"], np.sqrt(1.5))
        self.assertAlmostEqual(metrics["f_rmse"], 1.0)

    def test_loads_multiple_deepmd_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "type.raw").write_text("0\n1\n")
            (root / "type_map.raw").write_text("A B\n")
            for index in range(2):
                set_dir = root / f"set.{index:03d}"
                set_dir.mkdir()
                np.save(set_dir / "coord.npy", np.full((2, 6), index, dtype=float))
                np.save(set_dir / "box.npy", np.tile(np.eye(3).reshape(1, 9), (2, 1)))
                np.save(set_dir / "energy.npy", np.arange(2, dtype=float) + 2 * index)
                np.save(set_dir / "force.npy", np.zeros((2, 6)))
            data = load_deepmd(root, stride=2)
            self.assertEqual(data.coord.shape, (2, 2, 3))
            np.testing.assert_array_equal(data.energy, [0.0, 2.0])


if __name__ == "__main__":
    unittest.main()
