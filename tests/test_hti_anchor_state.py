import json
import shutil
import unittest
from pathlib import Path

from context import dpti


class TestHtiAnchorState(unittest.TestCase):
    def setUp(self):
        self.work_dir = Path("tmp_hti_anchor")
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
        self.work_dir.mkdir()

    def tearDown(self):
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)

    def test_get_npt_anchor_state(self):
        npt = self.work_dir / "npt"
        npt.mkdir()
        (npt / "jdata.json").write_text(json.dumps({"temp": 1600, "pres": 50000}))

        anchor = dpti.hti.get_npt_anchor_state(str(npt))

        self.assertEqual(anchor["t0"], 1600)
        self.assertEqual(anchor["p0"], 50000)

    def test_ti_prefers_anchor_position_from_hti_result(self):
        hti_result = {"t0": 1600, "p0": 50000}
        hti_input = {"temp": 1500, "pres": 10000}

        self.assertEqual(
            dpti.ti._get_hti_anchor_position("t", hti_result, hti_input), 1600
        )
        self.assertEqual(
            dpti.ti._get_hti_anchor_position("p", hti_result, hti_input), 50000
        )
        self.assertEqual(
            dpti.ti_water._get_hti_anchor_position("t", hti_result, hti_input), 1600
        )
        self.assertEqual(
            dpti.ti_water._get_hti_anchor_position("p", hti_result, hti_input), 50000
        )

    def test_ti_falls_back_to_hti_input_for_old_results(self):
        hti_result = {}
        hti_input = {"temp": 1500, "pres": 10000}

        self.assertEqual(
            dpti.ti._get_hti_anchor_position("t", hti_result, hti_input), 1500
        )
        self.assertEqual(
            dpti.ti._get_hti_anchor_position("p", hti_result, hti_input), 10000
        )
        self.assertEqual(
            dpti.ti_water._get_hti_anchor_position("t", hti_result, hti_input), 1500
        )
        self.assertEqual(
            dpti.ti_water._get_hti_anchor_position("p", hti_result, hti_input), 10000
        )


if __name__ == "__main__":
    unittest.main()
