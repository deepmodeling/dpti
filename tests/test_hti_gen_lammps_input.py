import os, textwrap
import numpy as np
import unittest
from context import deepti
# from potential_common import soft_param, soft_param_three_element, meam_model
# from deepti.lib.lammps import get_natoms, get_thermo, get_last_dump
# from deepti.lib.dump import from_system_data
from potential_common import soft_param
from deepti.hti import _gen_lammps_input
from numpy.testing import assert_almost_equal

class TestHtiGenLammpsInput(unittest.TestCase):
    def setUp(self) :
        self.maxDiff = None

    def test_normal(self):
        input = dict ( conf_file='conf.lmp', 
                        mass_map=[118.71],
                        lamb=0.075,
                        model="graph.pb",
                        m_spring_k=[2.3742,],
                        nsteps=500000,
                        dt=0.002,
                        ens='npt',
                        temp=400.0,
                        pres = 1.0, 
                        tau_t = 0.1,
                        tau_p = 0.5,
                        prt_freq = 100, 
                        copies = None,
                        crystal = 'vega', 
                        sparam = soft_param,
                        switch = 'three-step',
                        step = 'deep_on',
                        if_meam = False,
                        meam_model = None)
        lmp_str = _gen_lammps_input(**input)
        # with open()
        # print(lmp_str)

        