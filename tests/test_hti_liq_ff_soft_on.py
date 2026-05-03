import textwrap
import unittest

from potential_common import soft_param, soft_param_three_element

from dpti.hti_liq import _ff_soft_on


class TestFfSpring(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_one_element(self):
        input = {"lamb": 0.075, "sparam": soft_param}
        ret1 = textwrap.dedent(
            """\
        pair_style      lj/cut/soft 1.000000 0.500000 6.000000
        pair_coeff      1 1 0.030000 2.493672 0.500000
        fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_LAMBDA scale yes
        variable        EPSILON equal 0.030000
        compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_EPSILON
        variable        e_diff equal c_e_diff[1]
        """
        )
        ret2 = _ff_soft_on(**input)
        self.assertEqual(ret1, ret2)

    def test_three_element(self):
        input = {"lamb": 0.075, "sparam": soft_param_three_element}
        ret1 = textwrap.dedent(
            """\
        pair_style      lj/cut/soft 1.000000 0.600000 6.000000
        pair_coeff      1 1 0.030000 2.000000 0.500000
        pair_coeff      1 2 0.030000 2.010000 0.500000
        pair_coeff      1 3 0.030000 2.020000 0.500000
        pair_coeff      2 2 0.030000 2.110000 0.500000
        pair_coeff      2 3 0.030000 2.120000 0.500000
        pair_coeff      3 3 0.030000 2.220000 0.500000
        fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_LAMBDA scale yes
        variable        EPSILON equal 0.030000
        compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_EPSILON
        variable        e_diff equal c_e_diff[1]
        """
        )
        ret2 = _ff_soft_on(**input)
        self.assertEqual(ret1, ret2)

    def test_three_element_pair_epsilon(self):
        sparam = dict(soft_param_three_element)
        del sparam["epsilon"]
        sparam.update(
            {
                "epsilon_0_0": 0.03,
                "epsilon_0_1": 0.04,
                "epsilon_0_2": 0.05,
                "epsilon_1_1": 0.06,
                "epsilon_1_2": 0.07,
                "epsilon_2_2": 0.08,
            }
        )
        ret = _ff_soft_on(lamb=0.075, sparam=sparam)
        self.assertIn("variable        EPSILON_1_1 equal 0.030000\n", ret)
        self.assertIn("variable        EPSILON_3_3 equal 0.080000\n", ret)
        self.assertIn(
            "compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon 1 1 v_EPSILON_1_1",
            ret,
        )
        self.assertNotIn("/v_LAMBDA", ret)
