import os, textwrap
import numpy as np
import unittest
from context import dpti
# from potential_common import soft_param, soft_param_three_element, meam_model
# from dpti.lib.lammps import get_natoms, get_thermo, get_last_dump
# from dpti.lib.dump import from_system_data
from potential_common import soft_param, meam_model
from dpti.hti import _gen_lammps_input
from numpy.testing import assert_almost_equal
from unittest.mock import MagicMock, patch, PropertyMock

class TestHtiGenLammpsInput(unittest.TestCase):
    def setUp(self) :
        self.maxDiff = None

    @patch('numpy.random')
    def test_deepmd_lj_on(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        input = dict(conf_file='conf.lmp', 
                        mass_map=[118.71],
                        lamb=0.075,
                        model="graph.pb",
                        m_spring_k=[2.3742,],
                        nsteps=500000,
                        timestep=0.002,
                        ens='npt',
                        temp=400.0,
                        pres = 1.0, 
                        tau_t = 0.1,
                        tau_p = 0.5,
                        thermo_freq = 100, 
                        copies = None,
                        crystal = 'vega', 
                        sparam = soft_param,
                        switch = 'three-step',
                        step = 'lj_on',
                        if_meam = False,
                        meam_model = None)
        ret1 = textwrap.dedent("""\
        clear
        # --------------------- VARIABLES-------------------------
        variable        NSTEPS          equal 500000
        variable        THERMO_FREQ     equal 100
        variable        DUMP_FREQ       equal 100
        variable        TEMP            equal 400.000000
        variable        PRES            equal 1.000000
        variable        TAU_T           equal 0.100000
        variable        TAU_P           equal 0.500000
        variable        LAMBDA          equal 7.5000000000e-02
        variable        INV_LAMBDA      equal 9.2500000000e-01
        # ---------------------- INITIALIZAITION ------------------
        units           metal
        boundary        p p p
        atom_style      atomic
        # --------------------- ATOM DEFINITION ------------------
        box             tilt large
        read_data       conf.lmp
        change_box      all triclinic
        mass            1 118.710000
        # --------------------- FORCE FIELDS ---------------------
        variable        EPSILON equal 0.030000
        pair_style      lj/cut/soft 1.000000 0.500000 6.000000
        pair_coeff      1 1 ${EPSILON} 2.493672 0.500000
        fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_LAMBDA scale yes
        compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_EPSILON
        group           type_1 type 1
        fix             l_spring_1 type_1 spring/self 2.3742000000e+00
        fix_modify      l_spring_1 energy yes
        variable        l_spring equal f_l_spring_1
        # --------------------- MD SETTINGS ----------------------
        neighbor        1.0 bin
        timestep        0.002
        thermo          ${THERMO_FREQ}
        compute         allmsd all msd
        thermo_style    custom step ke pe etotal enthalpy temp press vol v_l_spring c_e_diff[1] c_allmsd[*]
        thermo_modify   format 9 %.16e
        thermo_modify   format 10 %.16e
        # dump            1 all custom ${DUMP_FREQ} dump.hti id type x y z vx vy vz
        fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}
        # --------------------- INITIALIZE -----------------------
        velocity        all create ${TEMP} 7858
        group           first id 1
        fix             fc first recenter INIT INIT INIT
        fix             fm first momentum 1 linear 1 1 1
        velocity        first zero linear
        # --------------------- RUN ------------------------------
        run             ${NSTEPS}
        write_data      out.lmp
        """)
        ret2 = _gen_lammps_input(**input)
        # print(ret2)
        self.assertEqual(ret1, ret2)


    @patch('numpy.random')
    def test_deepmd_deep_on(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        input = dict ( conf_file='conf.lmp', 
                        mass_map=[118.71],
                        lamb=0.075,
                        model="graph.pb",
                        m_spring_k=[2.3742,],
                        nsteps=500000,
                        timestep=0.002,
                        ens='npt',
                        temp=400.0,
                        pres = 1.0, 
                        tau_t = 0.1,
                        tau_p = 0.5,
                        thermo_freq = 100, 
                        copies = None,
                        crystal = 'vega', 
                        sparam = soft_param,
                        switch = 'three-step',
                        step = 'deep_on',
                        if_meam = False,
                        meam_model = None)
        ret1 = textwrap.dedent("""\
        clear
        # --------------------- VARIABLES-------------------------
        variable        NSTEPS          equal 500000
        variable        THERMO_FREQ     equal 100
        variable        DUMP_FREQ       equal 100
        variable        TEMP            equal 400.000000
        variable        PRES            equal 1.000000
        variable        TAU_T           equal 0.100000
        variable        TAU_P           equal 0.500000
        variable        LAMBDA          equal 7.5000000000e-02
        variable        INV_LAMBDA      equal 9.2500000000e-01
        # ---------------------- INITIALIZAITION ------------------
        units           metal
        boundary        p p p
        atom_style      atomic
        # --------------------- ATOM DEFINITION ------------------
        box             tilt large
        read_data       conf.lmp
        change_box      all triclinic
        mass            1 118.710000
        # --------------------- FORCE FIELDS ---------------------
        variable        EPSILON equal 0.030000
        variable        ONE equal 1
        pair_style      hybrid/overlay deepmd graph.pb lj/cut/soft 1.000000 0.500000 6.000000
        pair_coeff      * * deepmd
        pair_coeff      1 1 lj/cut/soft ${EPSILON} 2.493672 0.500000
        fix             tot_pot all adapt/fep 0 pair deepmd scale * * v_LAMBDA
        compute         e_diff all fep ${TEMP} pair deepmd scale * * v_ONE
        group           type_1 type 1
        fix             l_spring_1 type_1 spring/self 2.3742000000e+00
        fix_modify      l_spring_1 energy yes
        variable        l_spring equal f_l_spring_1
        # --------------------- MD SETTINGS ----------------------
        neighbor        1.0 bin
        timestep        0.002
        thermo          ${THERMO_FREQ}
        compute         allmsd all msd
        thermo_style    custom step ke pe etotal enthalpy temp press vol v_l_spring c_e_diff[1] c_allmsd[*]
        thermo_modify   format 9 %.16e
        thermo_modify   format 10 %.16e
        # dump            1 all custom ${DUMP_FREQ} dump.hti id type x y z vx vy vz
        fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}
        # --------------------- INITIALIZE -----------------------
        velocity        all create ${TEMP} 7858
        group           first id 1
        fix             fc first recenter INIT INIT INIT
        fix             fm first momentum 1 linear 1 1 1
        velocity        first zero linear
        # --------------------- RUN ------------------------------
        run             ${NSTEPS}
        write_data      out.lmp
        """)
        ret2 = _gen_lammps_input(**input)
        self.assertEqual(ret1, ret2)

    @patch('numpy.random')
    def test_deepmd_lj_off(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        input = dict ( conf_file='conf.lmp', 
                        mass_map=[118.71],
                        lamb=0.075,
                        model="graph.pb",
                        m_spring_k=[2.3742,],
                        nsteps=500000,
                        timestep=0.002,
                        ens='npt',
                        temp=400.0,
                        pres = 1.0, 
                        tau_t = 0.1,
                        tau_p = 0.5,
                        thermo_freq = 100, 
                        copies = None,
                        crystal = 'vega', 
                        sparam = soft_param,
                        switch = 'three-step',
                        step = 'spring_off',
                        if_meam = False,
                        meam_model = None)
        ret1 = textwrap.dedent("""\
        clear
        # --------------------- VARIABLES-------------------------
        variable        NSTEPS          equal 500000
        variable        THERMO_FREQ     equal 100
        variable        DUMP_FREQ       equal 100
        variable        TEMP            equal 400.000000
        variable        PRES            equal 1.000000
        variable        TAU_T           equal 0.100000
        variable        TAU_P           equal 0.500000
        variable        LAMBDA          equal 7.5000000000e-02
        variable        INV_LAMBDA      equal 9.2500000000e-01
        # ---------------------- INITIALIZAITION ------------------
        units           metal
        boundary        p p p
        atom_style      atomic
        # --------------------- ATOM DEFINITION ------------------
        box             tilt large
        read_data       conf.lmp
        change_box      all triclinic
        mass            1 118.710000
        # --------------------- FORCE FIELDS ---------------------
        variable        EPSILON equal 0.030000
        variable        INV_EPSILON equal -${EPSILON}
        pair_style      hybrid/overlay deepmd graph.pb lj/cut/soft 1.000000 0.500000 6.000000
        pair_coeff      * * deepmd
        pair_coeff      1 1 lj/cut/soft ${EPSILON} 2.493672 0.500000
        fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_INV_LAMBDA scale yes
        compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_INV_EPSILON
        group           type_1 type 1
        fix             l_spring_1 type_1 spring/self 2.1961350000e+00
        fix_modify      l_spring_1 energy yes
        variable        l_spring equal f_l_spring_1
        # --------------------- MD SETTINGS ----------------------
        neighbor        1.0 bin
        timestep        0.002
        thermo          ${THERMO_FREQ}
        compute         allmsd all msd
        thermo_style    custom step ke pe etotal enthalpy temp press vol v_l_spring c_e_diff[1] c_allmsd[*]
        thermo_modify   format 9 %.16e
        thermo_modify   format 10 %.16e
        # dump            1 all custom ${DUMP_FREQ} dump.hti id type x y z vx vy vz
        fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}
        # --------------------- INITIALIZE -----------------------
        velocity        all create ${TEMP} 7858
        group           first id 1
        fix             fc first recenter INIT INIT INIT
        fix             fm first momentum 1 linear 1 1 1
        velocity        first zero linear
        # --------------------- RUN ------------------------------
        run             ${NSTEPS}
        write_data      out.lmp
        """)
        ret2 = _gen_lammps_input(**input)
        self.assertEqual(ret1, ret2)

    @patch('numpy.random')
    def test_meam_deep_on(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        input = dict ( conf_file='conf.lmp', 
                        mass_map=[118.71],
                        lamb=0.075,
                        model=None,
                        m_spring_k=[2.3742,],
                        nsteps=500000,
                        timestep=0.002,
                        ens='npt',
                        temp=400.0,
                        pres = 1.0, 
                        tau_t = 0.1,
                        tau_p = 0.5,
                        thermo_freq = 100, 
                        copies = None,
                        crystal = 'vega', 
                        sparam = soft_param,
                        switch = 'three-step',
                        step = 'deep_on',
                        if_meam = True,
                        meam_model = meam_model)
        ret1 = textwrap.dedent("""\
        clear
        # --------------------- VARIABLES-------------------------
        variable        NSTEPS          equal 500000
        variable        THERMO_FREQ     equal 100
        variable        DUMP_FREQ       equal 100
        variable        TEMP            equal 400.000000
        variable        PRES            equal 1.000000
        variable        TAU_T           equal 0.100000
        variable        TAU_P           equal 0.500000
        variable        LAMBDA          equal 7.5000000000e-02
        variable        INV_LAMBDA      equal 9.2500000000e-01
        # ---------------------- INITIALIZAITION ------------------
        units           metal
        boundary        p p p
        atom_style      atomic
        # --------------------- ATOM DEFINITION ------------------
        box             tilt large
        read_data       conf.lmp
        change_box      all triclinic
        mass            1 118.710000
        # --------------------- FORCE FIELDS ---------------------
        variable        EPSILON equal 0.030000
        variable        ONE equal 1
        pair_style      hybrid/overlay meam lj/cut/soft 1.000000 0.500000 6.000000
        pair_coeff      * * meam library_18Metals.meam Sn Sn_18Metals.meam Sn
        pair_coeff      1 1 lj/cut/soft ${EPSILON} 2.493672 0.500000
        fix             tot_pot all adapt/fep 0 pair meam scale * * v_LAMBDA
        compute         e_diff all fep ${TEMP} pair meam scale * * v_ONE
        group           type_1 type 1
        fix             l_spring_1 type_1 spring/self 2.3742000000e+00
        fix_modify      l_spring_1 energy yes
        variable        l_spring equal f_l_spring_1
        # --------------------- MD SETTINGS ----------------------
        neighbor        1.0 bin
        timestep        0.002
        thermo          ${THERMO_FREQ}
        compute         allmsd all msd
        thermo_style    custom step ke pe etotal enthalpy temp press vol v_l_spring c_e_diff[1] c_allmsd[*]
        thermo_modify   format 9 %.16e
        thermo_modify   format 10 %.16e
        # dump            1 all custom ${DUMP_FREQ} dump.hti id type x y z vx vy vz
        fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}
        # --------------------- INITIALIZE -----------------------
        velocity        all create ${TEMP} 7858
        group           first id 1
        fix             fc first recenter INIT INIT INIT
        fix             fm first momentum 1 linear 1 1 1
        velocity        first zero linear
        # --------------------- RUN ------------------------------
        run             ${NSTEPS}
        write_data      out.lmp
        """)
        ret2 = _gen_lammps_input(**input)
        # print(ret2)
        self.assertEqual(ret1, ret2)

    @patch('numpy.random')
    def test_meam_spring_off(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        input = dict ( conf_file='conf.lmp', 
                        mass_map=[118.71],
                        lamb=0.075,
                        model=None,
                        m_spring_k=[2.3742,],
                        nsteps=500000,
                        timestep=0.002,
                        ens='npt',
                        temp=400.0,
                        pres = 1.0, 
                        tau_t = 0.1,
                        tau_p = 0.5,
                        thermo_freq = 100, 
                        copies = None,
                        crystal = 'vega', 
                        sparam = soft_param,
                        switch = 'three-step',
                        step = 'spring_off',
                        if_meam = True,
                        meam_model = meam_model)
        ret1 = textwrap.dedent("""\
        clear
        # --------------------- VARIABLES-------------------------
        variable        NSTEPS          equal 500000
        variable        THERMO_FREQ     equal 100
        variable        DUMP_FREQ       equal 100
        variable        TEMP            equal 400.000000
        variable        PRES            equal 1.000000
        variable        TAU_T           equal 0.100000
        variable        TAU_P           equal 0.500000
        variable        LAMBDA          equal 7.5000000000e-02
        variable        INV_LAMBDA      equal 9.2500000000e-01
        # ---------------------- INITIALIZAITION ------------------
        units           metal
        boundary        p p p
        atom_style      atomic
        # --------------------- ATOM DEFINITION ------------------
        box             tilt large
        read_data       conf.lmp
        change_box      all triclinic
        mass            1 118.710000
        # --------------------- FORCE FIELDS ---------------------
        variable        EPSILON equal 0.030000
        variable        INV_EPSILON equal -${EPSILON}
        pair_style      hybrid/overlay meam lj/cut/soft 1.000000 0.500000 6.000000
        pair_coeff      * * meam library_18Metals.meam Sn Sn_18Metals.meam Sn
        pair_coeff      1 1 lj/cut/soft ${EPSILON} 2.493672 0.500000
        fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_INV_LAMBDA scale yes
        compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_INV_EPSILON
        group           type_1 type 1
        fix             l_spring_1 type_1 spring/self 2.1961350000e+00
        fix_modify      l_spring_1 energy yes
        variable        l_spring equal f_l_spring_1
        # --------------------- MD SETTINGS ----------------------
        neighbor        1.0 bin
        timestep        0.002
        thermo          ${THERMO_FREQ}
        compute         allmsd all msd
        thermo_style    custom step ke pe etotal enthalpy temp press vol v_l_spring c_e_diff[1] c_allmsd[*]
        thermo_modify   format 9 %.16e
        thermo_modify   format 10 %.16e
        # dump            1 all custom ${DUMP_FREQ} dump.hti id type x y z vx vy vz
        fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}
        # --------------------- INITIALIZE -----------------------
        velocity        all create ${TEMP} 7858
        group           first id 1
        fix             fc first recenter INIT INIT INIT
        fix             fm first momentum 1 linear 1 1 1
        velocity        first zero linear
        # --------------------- RUN ------------------------------
        run             ${NSTEPS}
        write_data      out.lmp
        """)
        ret2 = _gen_lammps_input(**input)
        # print(ret2)
        self.assertEqual(ret1, ret2)

    def test_raise_err(self):
        input = dict ( conf_file='conf.lmp', 
                        mass_map=[118.71],
                        lamb=0.075,
                        model="graph.pb",
                        m_spring_k=[2.3742,],
                        nsteps=500000,
                        timestep=0.002,
                        ens='npt',
                        temp=400.0,
                        pres = 1.0, 
                        tau_t = 0.1,
                        tau_p = 0.5,
                        thermo_freq = 100, 
                        copies = None,
                        crystal = 'vega', 
                        sparam = soft_param,
                        switch = 'three-step',
                        step = 'spring_off',
                        if_meam = False,
                        meam_model = None)
        
        input2 = input.copy()
        input2['step'] = 'foo'
        with self.assertRaises(RuntimeError):
            _gen_lammps_input(**input2)

        
        input3 = input.copy()
        input3['switch'] = 'bar'
        with self.assertRaises(RuntimeError):
            _gen_lammps_input(**input3)

        input4 = input.copy()
        input4['ens'] = 'baz'
        with self.assertRaises(RuntimeError):
            _gen_lammps_input(**input4)

