import os, textwrap
import numpy as np
import unittest
from context import deepti
# print(deepti.equi)

class TestEquiForceField(unittest.TestCase):
    def setUp(self) :
        self.maxDiff = None

    # def test_gen_equi_force_field_model_wrong_type(self):
        # equi_settings = {'model': "meaningless_information"}
        # equi_settings = dict(model=)
        # with self.assertRaises(AssertionError):
        #     deepti.equi.gen_equi_force_field(equi_settings=equi_settings)
    
    def test_gen_equi_force_field_model_type_wrong_value(self):
        # equi_settings = {'model':{'deepmd_type': 'wrong_type'}}
        equi_settings = dict(model_type='wrong_type')
        with self.assertRaises(ValueError):
            deepti.equi.gen_equi_force_field(equi_settings=equi_settings)

    def test_gen_equi_force_field_deepmd(self):
        equi_settings = dict(model_type='deepmd', deepmd_model='graph.pb')
        # equi_settings = dict(model='graph.pb', if_meam=False)

        ret1 = textwrap.dedent("""\
        # --------------------- FORCE FIELDS ---------------------
        pair_style      deepmd graph.pb
        pair_coeff
        """)
        # ret1 = textwrap.dedent(ret1_raw)
        ret2 = deepti.equi.gen_equi_force_field(equi_settings=equi_settings)
        self.assertEqual(ret1, ret2)

    def test_gen_equi_force_field_meam(self):
        equi_settings = dict(model_type='meam',
            meam_element='Sn', 
            meam_library='library_18Metals.meam', 
            meam_potential='Sn_18Metals.meam', 
        )
        ret1 = textwrap.dedent("""\
        # --------------------- FORCE FIELDS ---------------------
        pair_style      meam
        pair_coeff      * * library_18Metals.meam Sn Sn_18Metals.meam Sn
        """)
        # ret1 = textwrap.dedent(ret1_raw)
        ret2 = deepti.equi.gen_equi_force_field(equi_settings=equi_settings)
        self.assertEqual(ret1, ret2)

    def test_gen_equi_force_field_meam_error(self):
        equi_settings2 = dict(model_type='meam',
            meam_element='Sn', 
            # meam_library='library_18Metals.meam', 
            meam_potential='Sn_18Metals.meam')
        with self.assertRaises(AssertionError):
            deepti.equi.gen_equi_force_field(equi_settings=equi_settings2)

        equi_settings3 = dict(model_type='meam',
            # meam_element='Sn', 
            meam_library='library_18Metals.meam', 
            meam_potential='Sn_18Metals.meam')
        with self.assertRaises(AssertionError):
            deepti.equi.gen_equi_force_field(equi_settings=equi_settings3)

        equi_settings4 = dict(model_type='meam',
            meam_element='Sn', 
            meam_library='library_18Metals.meam', 
            # meam_potential='Sn_18Metals.meam', 
            )
        with self.assertRaises(AssertionError):
            deepti.equi.gen_equi_force_field(equi_settings=equi_settings4)


if __name__ == '__main__':
    unittest.main()
