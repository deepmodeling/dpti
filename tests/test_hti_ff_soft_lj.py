import os, textwrap
import numpy as np
import unittest
from context import dpti
from potential_common import soft_param, soft_param_three_element, meam_model

class TestFfSpring(unittest.TestCase):
    def setUp(self) :
        self.maxDiff = None

    def test_lj_on(self):
        input = dict(lamb=0.075,
            model="graph.pb",
            m_spring_k=[118.71],
            step="lj_on",
            sparam=soft_param,
            if_meam=False,
            meam_model=None
        )
        ret1 = textwrap.dedent("""\
        # --------------------- FORCE FIELDS ---------------------
        variable        EPSILON equal 0.030000
        pair_style      lj/cut/soft 1.000000 0.500000 6.000000
        pair_coeff      1 1 ${EPSILON} 2.493672 0.500000
        fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_LAMBDA scale yes
        compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_EPSILON
        group           type_1 type 1
        fix             l_spring_1 type_1 spring/self 1.1871000000e+02
        fix_modify      l_spring_1 energy yes
        variable        l_spring equal f_l_spring_1
        """)
        ret2 = dpti.hti._ff_soft_lj(**input)
        # print('--------')
        # # print(ret1)
        # print('--------')
        # # print(ret2)
        # print('--------')
        self.assertEqual(ret1, ret2)

    def test_deep_on(self):
        input = dict(lamb=0.075,
            model="graph.pb",
            m_spring_k=[118.71],
            step="deep_on",
            sparam=soft_param,
            if_meam=False,
            meam_model=None
        )
        ret1 = textwrap.dedent("""\
        # --------------------- FORCE FIELDS ---------------------
        variable        EPSILON equal 0.030000
        variable        ONE equal 1
        pair_style      hybrid/overlay deepmd graph.pb lj/cut/soft 1.000000 0.500000 6.000000
        pair_coeff      * * deepmd
        pair_coeff      1 1 lj/cut/soft ${EPSILON} 2.493672 0.500000
        fix             tot_pot all adapt/fep 0 pair deepmd scale * * v_LAMBDA
        compute         e_diff all fep ${TEMP} pair deepmd scale * * v_ONE
        group           type_1 type 1
        fix             l_spring_1 type_1 spring/self 1.1871000000e+02
        fix_modify      l_spring_1 energy yes
        variable        l_spring equal f_l_spring_1
        """)
        ret2 = dpti.hti._ff_soft_lj(**input)
        # print('--------')
        # print(ret1)
        # print('--------')
        # print(ret2)
        # print('--------')
        self.assertEqual(ret1, ret2)

    def test_spring_off(self):
        input = dict(lamb=0.075,
            model="graph.pb",
            m_spring_k=[118.71],
            step="spring_off",
            sparam=soft_param,
            if_meam=False,
            meam_model=None
        )
        ret1 = textwrap.dedent("""\
        # --------------------- FORCE FIELDS ---------------------
        variable        EPSILON equal 0.030000
        variable        INV_EPSILON equal -${EPSILON}
        pair_style      hybrid/overlay deepmd graph.pb lj/cut/soft 1.000000 0.500000 6.000000
        pair_coeff      * * deepmd
        pair_coeff      1 1 lj/cut/soft ${EPSILON} 2.493672 0.500000
        fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_INV_LAMBDA scale yes
        compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_INV_EPSILON
        group           type_1 type 1
        fix             l_spring_1 type_1 spring/self 1.0980675000e+02
        fix_modify      l_spring_1 energy yes
        variable        l_spring equal f_l_spring_1
        """)
        ret2 = dpti.hti._ff_soft_lj(**input)
        self.assertEqual(ret1, ret2)

# def test_me
    def test_meam_deep_on(self):
        input = dict(lamb=0.075,
            model="graph.pb",
            m_spring_k=[118.71],
            step="deep_on",
            sparam=soft_param,
            if_meam=True,
            meam_model=meam_model
        )
        ret1 = textwrap.dedent("""\
        # --------------------- FORCE FIELDS ---------------------
        variable        EPSILON equal 0.030000
        variable        ONE equal 1
        pair_style      hybrid/overlay meam lj/cut/soft 1.000000 0.500000 6.000000
        pair_coeff      * * meam library_18Metals.meam Sn Sn_18Metals.meam Sn
        pair_coeff      1 1 lj/cut/soft ${EPSILON} 2.493672 0.500000
        fix             tot_pot all adapt/fep 0 pair meam scale * * v_LAMBDA
        compute         e_diff all fep ${TEMP} pair meam scale * * v_ONE
        group           type_1 type 1
        fix             l_spring_1 type_1 spring/self 1.1871000000e+02
        fix_modify      l_spring_1 energy yes
        variable        l_spring equal f_l_spring_1
        """)
        ret2 = dpti.hti._ff_soft_lj(**input)
        # print('--------')
        # print(ret1)
        # print('--------')
        # print(ret2)
        # print('--------')
        self.assertEqual(ret1, ret2)

    def test_meam_spring_off(self):
        input = dict(lamb=0.075,
            model="graph.pb",
            m_spring_k=[118.71],
            step="spring_off",
            sparam=soft_param,
            if_meam=True,
            meam_model=meam_model
        )
        
        ret1 = textwrap.dedent("""\
        # --------------------- FORCE FIELDS ---------------------
        variable        EPSILON equal 0.030000
        variable        INV_EPSILON equal -${EPSILON}
        pair_style      hybrid/overlay meam lj/cut/soft 1.000000 0.500000 6.000000
        pair_coeff      * * meam library_18Metals.meam Sn Sn_18Metals.meam Sn
        pair_coeff      1 1 lj/cut/soft ${EPSILON} 2.493672 0.500000
        fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_INV_LAMBDA scale yes
        compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_INV_EPSILON
        group           type_1 type 1
        fix             l_spring_1 type_1 spring/self 1.0980675000e+02
        fix_modify      l_spring_1 energy yes
        variable        l_spring equal f_l_spring_1
        """)
        ret2 = dpti.hti._ff_soft_lj(**input)
        # print('--------')
        # print(ret1)
        # print('--------')
        # print(ret2)
        # print('--------')
        self.assertEqual(ret1, ret2)


#     def test_spring_var_spring(self):
#         input = dict(lamb=0.075, m_spring_k=[118.71],
#             var_spring=True)
#         ret1 = textwrap.dedent("""\
#         group           type_1 type 1
#         fix             l_spring_1 type_1 spring/self 1.0980675000e+02
#         fix_modify      l_spring_1 energy yes
#         variable        l_spring equal f_l_spring_1
#         """)
#         ret2 = dpti.hti._ff_spring(**input)
#         self.assertEqual(ret1, ret2)

#     def test_spring_multiple_element(self):
#         input = dict(lamb=0.075, m_spring_k=[118.71, 207.2],
#             var_spring=False)
#         ret1 = textwrap.dedent("""\
#         group           type_1 type 1
#         group           type_2 type 2
#         fix             l_spring_1 type_1 spring/self 1.1871000000e+02
#         fix_modify      l_spring_1 energy yes
#         fix             l_spring_2 type_2 spring/self 2.0720000000e+02
#         fix_modify      l_spring_2 energy yes
#         variable        l_spring equal f_l_spring_1+f_l_spring_2
#         """)
#         ret2 = dpti.hti._ff_spring(**input)
#         self.assertEqual(ret1, ret2)

# def test_spring_var_spring_multiple_element(self):
#         input = dict(lamb=0.20, m_spring_k=[118.71, 207.2],
#             var_spring=False)
#         ret1 = textwrap.dedent("""\
#         group           type_1 type 1
#         group           type_2 type 2
#         fix             l_spring_1 type_1 spring/self 9.4968000000e+01
#         fix_modify      l_spring_1 energy yes
#         fix             l_spring_2 type_2 spring/self 1.6576000000e+02
#         fix_modify      l_spring_2 energy yes
#         variable        l_spring equal f_l_spring_1+f_l_spring_2
#         """)
#         ret2 = dpti.hti._ff_spring(**input)
#         self.assertEqual(ret1, ret2)


if __name__ == '__main__':
    unittest.main()
