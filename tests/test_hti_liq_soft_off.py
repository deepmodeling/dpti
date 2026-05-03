import textwrap
import unittest

from potential_common import meam_model, soft_param, soft_param_three_element

from dpti.hti_liq import _ff_soft_off


class TestSoftOff(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_one_element(self):
        input = {
            "lamb": 0.075,
            "sparam": soft_param,
            "model": "graph.pb",
            "if_meam": False,
            "meam_model": None,
        }
        ret1 = textwrap.dedent(
            """\
        variable        INV_LAMBDA equal 1-${LAMBDA}
        pair_style      hybrid/overlay deepmd graph.pb lj/cut/soft 1.000000 0.500000 6.000000
        pair_coeff      * * deepmd
        pair_coeff      1 1 lj/cut/soft 0.030000 2.493672 0.500000
        fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_INV_LAMBDA scale yes
        variable        EPSILON equal -0.030000
        compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_EPSILON
        variable        e_diff equal c_e_diff[1]
        """
        )
        ret2 = _ff_soft_off(**input)
        self.assertEqual(ret1, ret2)

    def test_three_element(self):
        input = {
            "lamb": 0.075,
            "sparam": soft_param_three_element,
            "model": "graph.pb",
            "if_meam": False,
            "meam_model": None,
        }
        ret1 = textwrap.dedent(
            """\
        variable        INV_LAMBDA equal 1-${LAMBDA}
        pair_style      hybrid/overlay deepmd graph.pb lj/cut/soft 1.000000 0.600000 6.000000
        pair_coeff      * * deepmd
        pair_coeff      1 1 lj/cut/soft 0.030000 2.000000 0.500000
        pair_coeff      1 2 lj/cut/soft 0.030000 2.010000 0.500000
        pair_coeff      1 3 lj/cut/soft 0.030000 2.020000 0.500000
        pair_coeff      2 2 lj/cut/soft 0.030000 2.110000 0.500000
        pair_coeff      2 3 lj/cut/soft 0.030000 2.120000 0.500000
        pair_coeff      3 3 lj/cut/soft 0.030000 2.220000 0.500000
        fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_INV_LAMBDA scale yes
        variable        EPSILON equal -0.030000
        compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_EPSILON
        variable        e_diff equal c_e_diff[1]
        """
        )
        ret2 = _ff_soft_off(**input)
        self.assertEqual(ret1, ret2)

    def test_deepmd(self):
        input = {
            "lamb": 0.075,
            "sparam": soft_param,
            "model": "graph.pb",
            "if_meam": False,
            "meam_model": None,
        }
        ret1 = textwrap.dedent(
            """\
        variable        INV_LAMBDA equal 1-${LAMBDA}
        pair_style      hybrid/overlay deepmd graph.pb lj/cut/soft 1.000000 0.500000 6.000000
        pair_coeff      * * deepmd
        pair_coeff      1 1 lj/cut/soft 0.030000 2.493672 0.500000
        fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_INV_LAMBDA scale yes
        variable        EPSILON equal -0.030000
        compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_EPSILON
        variable        e_diff equal c_e_diff[1]
        """
        )
        ret2 = _ff_soft_off(**input)
        self.assertEqual(ret1, ret2)

    def test_meam(self):
        input = {
            "lamb": 0.075,
            "sparam": soft_param,
            "model": None,
            "if_meam": True,
            "meam_model": meam_model,
        }
        ret1 = textwrap.dedent(
            """\
        variable        INV_LAMBDA equal 1-${LAMBDA}
        pair_style      hybrid/overlay meam lj/cut/soft 1.000000 0.500000 6.000000
        pair_coeff      * * meam library_18Metals.meam Sn Sn_18Metals.meam Sn
        pair_coeff      1 1 lj/cut/soft 0.030000 2.493672 0.500000
        fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_INV_LAMBDA scale yes
        variable        EPSILON equal -0.030000
        compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_EPSILON
        variable        e_diff equal c_e_diff[1]
        """
        )
        ret2 = _ff_soft_off(**input)
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
        ret = _ff_soft_off(
            lamb=0.075,
            sparam=sparam,
            model="graph.pb",
            if_meam=False,
            meam_model=None,
        )
        self.assertIn("variable        EPSILON_1_1 equal -0.030000\n", ret)
        self.assertIn("variable        EPSILON_3_3 equal -0.080000\n", ret)
        self.assertIn(
            "compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon 1 1 v_EPSILON_1_1",
            ret,
        )
        self.assertNotIn("/v_INV_LAMBDA", ret)
