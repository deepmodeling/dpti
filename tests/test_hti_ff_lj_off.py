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
        input = dict(lamb=0.075, model="graph.pb", sparam=soft_param,
            if_meam=False, meam_model=meam_model)
        ret1 = textwrap.dedent("""\
        variable        EPSILON equal 0.030000
        variable        INV_EPSILON equal -${EPSILON}
        pair_style      hybrid/overlay deepmd graph.pb lj/cut/soft 1.000000 0.500000 6.000000
        pair_coeff      * * deepmd
        pair_coeff      1 1 lj/cut/soft ${EPSILON} 2.493672 0.500000
        fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_INV_LAMBDA scale yes
        compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_INV_EPSILON
        """)
        # ret1 = textwrap.dedent(ret1_raw)
        ret2 = dpti.hti._ff_lj_off(**input)
        # print('--------')
        # print(ret1)
        # print('--------')
        # print(ret2)
        # print('--------')
        self.assertEqual(ret1, ret2)

    def test_meam(self):
        input = dict(lamb=0.075, model="graph.pb", sparam=soft_param,
            if_meam=True, meam_model=meam_model)

    #     input = dict(lamb=0.075, model=None, sparam=soft_param,
    #         if_meam=True, meam_model=meam_model)
        ret1 = textwrap.dedent("""\
        variable        EPSILON equal 0.030000
        variable        INV_EPSILON equal -${EPSILON}
        pair_style      hybrid/overlay meam lj/cut/soft 1.000000 0.500000 6.000000
        pair_coeff      * * meam library_18Metals.meam Sn Sn_18Metals.meam Sn
        pair_coeff      1 1 lj/cut/soft ${EPSILON} 2.493672 0.500000
        fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_INV_LAMBDA scale yes
        compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_INV_EPSILON
        """)
        ret2 = dpti.hti._ff_lj_off(**input)
        # print('--------')
        # print(ret1)
        # print('--------')
        # print(ret2)
        # print('--------')
        self.assertEqual(ret1, ret2)



if __name__ == '__main__':
    unittest.main()
