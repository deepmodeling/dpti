import os, textwrap
import numpy as np
import unittest
from context import dpti
from potential_common import meam_model

class TestEquiForceField(unittest.TestCase):
    def setUp(self) :
        self.maxDiff = None

    def test_deepmd(self):
        input = dict(model="graph.pb", if_meam=False, meam_model=None)

        ret1 = textwrap.dedent("""\
        # --------------------- FORCE FIELDS ---------------------
        pair_style      deepmd graph.pb
        pair_coeff
        """)
        ret2 = dpti.equi.gen_equi_force_field(**input)
        self.assertEqual(ret1, ret2)

    def test_meam(self):
        input = dict(model=None,
            if_meam=True,
            meam_model=meam_model
        )
        ret1 = textwrap.dedent("""\
        # --------------------- FORCE FIELDS ---------------------
        pair_style      meam
        pair_coeff      * * library_18Metals.meam Sn Sn_18Metals.meam Sn
        """)
        ret2 = dpti.equi.gen_equi_force_field(**input)
        self.assertEqual(ret1, ret2)

    def test_meam_raise_error(self):
        input = dict(model=None,
            if_meam=True,
            meam_model={})
        with self.assertRaises(KeyError):
            dpti.equi.gen_equi_force_field(**input)

    def test_meam_raise_error_lack_key(self):
        err_meam_model1 = dict(
            meam_library='library_18Metals.meam', 
            meam_potential='Sn_18Metals.meam', 
        )
        input = dict(model=None,
            if_meam=True,
            meam_model=err_meam_model1)
        with self.assertRaises(KeyError):
            dpti.equi.gen_equi_force_field(**input)

if __name__ == '__main__':
    unittest.main()
