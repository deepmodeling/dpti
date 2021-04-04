import os, textwrap
import numpy as np
import unittest
from context import dpti
from potential_common import soft_param, soft_param_three_element, meam_model
# from potential_common import 
# print(dpti.equi)

class TestEquiForceField(unittest.TestCase):
    def setUp(self) :
        self.maxDiff = None

    def test_deepmd(self):
        input = dict(lamb=0.075, model="graph.pb", sparam=soft_param)
        ret1 = textwrap.dedent("""\
        variable        EPSILON equal 0.030000
        variable        ONE equal 1
        pair_style      hybrid/overlay deepmd graph.pb lj/cut/soft 1.000000 0.500000 6.000000
        pair_coeff      * * deepmd
        pair_coeff      1 1 lj/cut/soft ${EPSILON} 2.493672 0.500000
        fix             tot_pot all adapt/fep 0 pair deepmd scale * * v_LAMBDA
        compute         e_diff all fep ${TEMP} pair deepmd scale * * v_ONE
        """)
        # ret1 = textwrap.dedent(ret1_raw)
        ret2 = dpti.hti._ff_deep_on(**input)
        # print('--------')
        # print(ret1)
        # print('--------')
        # print(ret2)
        # print('--------')
        self.assertEqual(ret1, ret2)

    def test_meam(self):
        input = dict(lamb=0.075, model=None, sparam=soft_param,
            if_meam=True, meam_model=meam_model)
        ret1 = textwrap.dedent("""\
        variable        EPSILON equal 0.030000
        variable        ONE equal 1
        pair_style      hybrid/overlay meam lj/cut/soft 1.000000 0.500000 6.000000
        pair_coeff      * * meam library_18Metals.meam Sn Sn_18Metals.meam Sn
        pair_coeff      1 1 lj/cut/soft ${EPSILON} 2.493672 0.500000
        fix             tot_pot all adapt/fep 0 pair meam scale * * v_LAMBDA
        compute         e_diff all fep ${TEMP} pair meam scale * * v_ONE
        """)
        ret2 = dpti.hti._ff_deep_on(**input)
        # print('--------')
        # # print(ret1)
        # print('--------')
        # print(ret2)
        # print('--------')
        self.assertEqual(ret1, ret2)
    
    def test_meam_key_error(self):
        input = dict(lamb=0.075, model=None, sparam=soft_param,
            if_meam=True, meam_model={})
        with self.assertRaises(KeyError):
            dpti.hti._ff_deep_on(**input)



if __name__ == '__main__':
    unittest.main()
