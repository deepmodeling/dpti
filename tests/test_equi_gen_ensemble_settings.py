import os
import numpy as np
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from context import deepti

class TestEquiEnsembleSetting(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
    
    @patch('numpy.random')
    def test_gen_equi_ensemble_settings_nvt(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        ret1 = """fix             1 all nvt temp ${TEMP} ${TEMP} ${TAU_T}
fix             mzero all momentum 10 linear 1 1 1
# --------------------- INITIALIZE -----------------------
velocity        all create ${TEMP} 7858
velocity        all zero linear
# --------------------- RUN ------------------------------
run             ${NSTEPS}
write_data      out.lmp
"""
        ret2 = deepti.equi.gen_equi_ensemble_settings(equi_settings=dict(ens='nvt'))
        self.assertEqual(ret1, ret2)

    @patch('numpy.random')
    def test_gen_equi_ensemble_settings_npt(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        ret_ensemble_npt = "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}\n"
        ret_ensemble_npt_aniso = "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} aniso ${PRES} ${PRES} ${TAU_P}\n"
        ret_ensemble_npt_xy = "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} aniso ${PRES} ${PRES} ${TAU_P} couple xy\n"
        ret_ensemble_npt_tri = "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} tri ${PRES} ${PRES} ${TAU_P}\n"
        ret_ensemble_nve = "fix             1 all nve\n"
        ret_other = """fix             mzero all momentum 10 linear 1 1 1
# --------------------- INITIALIZE -----------------------
velocity        all create ${TEMP} 7858
velocity        all zero linear
# --------------------- RUN ------------------------------
run             ${NSTEPS}
write_data      out.lmp
"""
        ret_npt = ret_ensemble_npt + ret_other
        ret_npt_aniso = ret_ensemble_npt_aniso + ret_other
        ret_npt_xy = ret_ensemble_npt_xy + ret_other
        ret_npt_tri = ret_ensemble_npt_tri + ret_other
        ret_nve = ret_ensemble_nve + ret_other
        ret2 = deepti.equi.gen_equi_ensemble_settings(equi_settings=dict(ens='npt'))
        ret3 = deepti.equi.gen_equi_ensemble_settings(equi_settings=dict(ens='npt-iso'))
        ret4 = deepti.equi.gen_equi_ensemble_settings(equi_settings=dict(ens='npt-aniso'))
        ret5 = deepti.equi.gen_equi_ensemble_settings(equi_settings=dict(ens='npt-xy'))
        ret6 = deepti.equi.gen_equi_ensemble_settings(equi_settings=dict(ens='npt-tri'))
        ret7 = deepti.equi.gen_equi_ensemble_settings(equi_settings=dict(ens='nve'))
        self.assertEqual(ret_npt, ret2)
        self.assertEqual(ret_npt, ret3)
        self.assertEqual(ret_npt_aniso, ret4)
        self.assertEqual(ret_npt_xy, ret5)
        self.assertEqual(ret_npt_tri, ret6)
        self.assertEqual(ret_nve, ret7)


if __name__ == '__main__':
    unittest.main()
