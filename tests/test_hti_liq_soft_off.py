import os, textwrap
import numpy as np
import unittest
from context import dpti
from potential_common import soft_param, soft_param_three_element, meam_model
from dpti.hti_liq import _ff_soft_off

class TestSoftOff(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_one_element(self):
        input = dict(lamb=0.075,
            sparam=soft_param, model="graph.pb",
            if_meam=False, meam_model=None)
        ret1 = textwrap.dedent("""\
        variable        INV_LAMBDA equal 1-${LAMBDA}
        variable        EPSILON equal 0.030000
        variable        INV_EPSILON equal -${EPSILON}
        pair_style      hybrid/overlay deepmd graph.pb lj/cut/soft 1.000000 0.500000 6.000000
        pair_coeff      * * deepmd
        pair_coeff      1 1 lj/cut/soft ${EPSILON} 2.493672 0.500000
        fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_INV_LAMBDA scale yes
        compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_INV_EPSILON
        """)
        ret2 = _ff_soft_off(**input)
        self.assertEqual(ret1, ret2)
    
    def test_three_element(self):
        input = dict(lamb=0.075,
            sparam=soft_param_three_element, model="graph.pb",
            if_meam=False, meam_model=None)
        ret1 = textwrap.dedent("""\
        variable        INV_LAMBDA equal 1-${LAMBDA}
        variable        EPSILON equal 0.030000
        variable        INV_EPSILON equal -${EPSILON}
        pair_style      hybrid/overlay deepmd graph.pb lj/cut/soft 1.000000 0.600000 6.000000
        pair_coeff      * * deepmd
        pair_coeff      1 1 lj/cut/soft ${EPSILON} 2.000000 0.500000
        pair_coeff      1 2 lj/cut/soft ${EPSILON} 2.010000 0.500000
        pair_coeff      1 3 lj/cut/soft ${EPSILON} 2.020000 0.500000
        pair_coeff      2 2 lj/cut/soft ${EPSILON} 2.110000 0.500000
        pair_coeff      2 3 lj/cut/soft ${EPSILON} 2.120000 0.500000
        pair_coeff      3 3 lj/cut/soft ${EPSILON} 2.220000 0.500000
        fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_INV_LAMBDA scale yes
        compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_INV_EPSILON
        """)
        ret2 = _ff_soft_off(**input)
        self.assertEqual(ret1, ret2)

    def test_deepmd(self):
        input = dict(lamb=0.075,
            sparam=soft_param, model="graph.pb",
            if_meam=False, meam_model=None)
        ret1 = textwrap.dedent("""\
        variable        INV_LAMBDA equal 1-${LAMBDA}
        variable        EPSILON equal 0.030000
        variable        INV_EPSILON equal -${EPSILON}
        pair_style      hybrid/overlay deepmd graph.pb lj/cut/soft 1.000000 0.500000 6.000000
        pair_coeff      * * deepmd
        pair_coeff      1 1 lj/cut/soft ${EPSILON} 2.493672 0.500000
        fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_INV_LAMBDA scale yes
        compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_INV_EPSILON
        """)
        ret2 = _ff_soft_off(**input)
        self.assertEqual(ret1, ret2)
    
    def test_meam(self):
        input = dict(lamb=0.075,
            sparam=soft_param, model=None,
            if_meam=True, meam_model=meam_model)
        ret1 = textwrap.dedent("""\
        variable        INV_LAMBDA equal 1-${LAMBDA}
        variable        EPSILON equal 0.030000
        variable        INV_EPSILON equal -${EPSILON}
        pair_style      hybrid/overlay meam lj/cut/soft 1.000000 0.500000 6.000000
        pair_coeff      * * meam library_18Metals.meam Sn Sn_18Metals.meam Sn
        pair_coeff      1 1 lj/cut/soft ${EPSILON} 2.493672 0.500000
        fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_INV_LAMBDA scale yes
        compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_INV_EPSILON
        """)
        ret2 = _ff_soft_off(**input)
        self.assertEqual(ret1, ret2)
