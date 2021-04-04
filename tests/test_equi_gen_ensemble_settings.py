import os, textwrap
import numpy as np
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from context import dpti

class TestEquiEnsembleSetting(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
    
    @patch('numpy.random')
    def test_gen_equi_ensemble_settings_nvt(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        input = dict(ens='nvt')
        ret1 = textwrap.dedent("""\
        fix             1 all nvt temp ${TEMP} ${TEMP} ${TAU_T}
        fix             mzero all momentum 10 linear 1 1 1
        # --------------------- INITIALIZE -----------------------
        velocity        all create ${TEMP} 7858
        velocity        all zero linear
        # --------------------- RUN ------------------------------
        run             ${NSTEPS}
        write_data      out.lmp
        """)
        ret2 = dpti.equi.gen_equi_ensemble_settings(**input)
        self.assertEqual(ret1, ret2)

    @patch('numpy.random')
    def test_gen_equi_ensemble_settings_npt(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        ret_ensemble_npt = "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}\n"
        ret_ensemble_npt_aniso = "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} aniso ${PRES} ${PRES} ${TAU_P}\n"
        ret_ensemble_npt_xy = "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} aniso ${PRES} ${PRES} ${TAU_P} couple xy\n"
        ret_ensemble_npt_tri = "fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} tri ${PRES} ${PRES} ${TAU_P}\n"
        ret_ensemble_nve = "fix             1 all nve\n"
        ret_other = textwrap.dedent("""\
        fix             mzero all momentum 10 linear 1 1 1
        # --------------------- INITIALIZE -----------------------
        velocity        all create ${TEMP} 7858
        velocity        all zero linear
        # --------------------- RUN ------------------------------
        run             ${NSTEPS}
        write_data      out.lmp
        """)
        ret_npt = ret_ensemble_npt + ret_other
        ret_npt_aniso = ret_ensemble_npt_aniso + ret_other
        ret_npt_xy = ret_ensemble_npt_xy + ret_other
        ret_npt_tri = ret_ensemble_npt_tri + ret_other
        ret_nve = ret_ensemble_nve + ret_other
        ret2 = dpti.equi.gen_equi_ensemble_settings(ens='npt')
        ret3 = dpti.equi.gen_equi_ensemble_settings(ens='npt-iso')
        ret4 = dpti.equi.gen_equi_ensemble_settings(ens='npt-aniso')
        ret5 = dpti.equi.gen_equi_ensemble_settings(ens='npt-xy')
        ret6 = dpti.equi.gen_equi_ensemble_settings(ens='npt-tri')
        ret7 = dpti.equi.gen_equi_ensemble_settings(ens='nve')
        self.assertEqual(ret_npt, ret2)
        self.assertEqual(ret_npt, ret3)
        self.assertEqual(ret_npt_aniso, ret4)
        self.assertEqual(ret_npt_xy, ret5)
        self.assertEqual(ret_npt_tri, ret6)
        self.assertEqual(ret_nve, ret7)
        with self.assertRaises(RuntimeError):
            dpti.equi.gen_equi_ensemble_settings(ens='foo')


if __name__ == '__main__':
    unittest.main()
