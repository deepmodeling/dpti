import os, textwrap
import numpy as np
import unittest
from context import dpti
from potential_common import soft_param, soft_param_three_element, meam_model

class TestHtiFfSpring(unittest.TestCase):
    def setUp(self) :
        self.maxDiff = None

    def test_spring_not_var_spring(self):
        input = dict(lamb=0.075, m_spring_k=[118.71],
            var_spring=False)
        ret1 = textwrap.dedent("""\
        group           type_1 type 1
        fix             l_spring_1 type_1 spring/self 1.1871000000e+02
        fix_modify      l_spring_1 energy yes
        variable        l_spring equal f_l_spring_1
        """)
        ret2 = dpti.hti._ff_spring(**input)
        self.assertEqual(ret1, ret2)

    def test_spring_var_spring(self):
        input = dict(lamb=0.075, m_spring_k=[118.71],
            var_spring=True)
        ret1 = textwrap.dedent("""\
        group           type_1 type 1
        fix             l_spring_1 type_1 spring/self 1.0980675000e+02
        fix_modify      l_spring_1 energy yes
        variable        l_spring equal f_l_spring_1
        """)
        ret2 = dpti.hti._ff_spring(**input)
        self.assertEqual(ret1, ret2)

    def test_spring_multiple_element(self):
        input = dict(lamb=0.075, m_spring_k=[118.71, 207.2],
            var_spring=False)
        ret1 = textwrap.dedent("""\
        group           type_1 type 1
        group           type_2 type 2
        fix             l_spring_1 type_1 spring/self 1.1871000000e+02
        fix_modify      l_spring_1 energy yes
        fix             l_spring_2 type_2 spring/self 2.0720000000e+02
        fix_modify      l_spring_2 energy yes
        variable        l_spring equal f_l_spring_1+f_l_spring_2
        """)
        ret2 = dpti.hti._ff_spring(**input)
        self.assertEqual(ret1, ret2)

def test_spring_var_spring_multiple_element(self):
        input = dict(lamb=0.20, m_spring_k=[118.71, 207.2],
            var_spring=False)
        ret1 = textwrap.dedent("""\
        group           type_1 type 1
        group           type_2 type 2
        fix             l_spring_1 type_1 spring/self 9.4968000000e+01
        fix_modify      l_spring_1 energy yes
        fix             l_spring_2 type_2 spring/self 1.6576000000e+02
        fix_modify      l_spring_2 energy yes
        variable        l_spring equal f_l_spring_1+f_l_spring_2
        """)
        ret2 = dpti.hti._ff_spring(**input)
        self.assertEqual(ret1, ret2)


if __name__ == '__main__':
    unittest.main()
