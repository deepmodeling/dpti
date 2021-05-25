import os, json, shutil, textwrap
import numpy as np
import unittest
from context import dpti
from unittest.mock import MagicMock, patch, PropertyMock
from dpti.lib.utils import get_file_md5
from dpti import ti
from potential_common import meam_model


class TestTiGenLammpsInput(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    @patch('numpy.random')
    def test_deepmd(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        input = dict(
            conf_file='conf.lmp', 
            mass_map=[118.71,],
            model="graph.pb",
            nsteps=200000,
            timestep=0.002,
            ens='npt',
            temp=200,
            pres=50000, 
            tau_t=0.1,
            tau_p=0.5,
            thermo_freq=10,
            copies=None,
            if_meam=False,
            meam_model=None
        )
        ret1 = textwrap.dedent("""\
        clear
        # --------------------- VARIABLES-------------------------
        variable        NSTEPS          equal 200000
        variable        THERMO_FREQ     equal 10
        variable        TEMP            equal 200.000000
        variable        PRES            equal 50000.000000
        variable        TAU_T           equal 0.100000
        variable        TAU_P           equal 0.500000
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
        pair_style      deepmd graph.pb
        pair_coeff
        # --------------------- MD SETTINGS ----------------------
        neighbor        1.0 bin
        timestep        0.002
        thermo          ${THERMO_FREQ}
        compute         allmsd all msd
        thermo_style    custom step ke pe etotal enthalpy temp press vol c_allmsd[*]
        # dump            1 all custom ${DUMP_FREQ} traj.dump id type x y z
        fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}
        fix             mzero all momentum 10 linear 1 1 1
        # --------------------- INITIALIZE -----------------------
        velocity        all create ${TEMP} 7858
        velocity        all zero linear
        # --------------------- RUN ------------------------------
        run             ${NSTEPS}
        write_data      out.lmp
        """)
        ret2 = ti._gen_lammps_input(**input)
        self.assertEqual(ret1, ret2)

    @patch('numpy.random')
    def test_meam(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        input = dict(
            conf_file='conf.lmp', 
            mass_map=[118.71,],
            model="graph.pb",
            nsteps=200000,
            timestep=0.002,
            ens='npt',
            temp=200,
            pres=50000, 
            tau_t=0.1,
            tau_p=0.5,
            thermo_freq=10,
            copies=None,
            if_meam=True,
            meam_model=meam_model
        )
        ret1 = textwrap.dedent("""\
        clear
        # --------------------- VARIABLES-------------------------
        variable        NSTEPS          equal 200000
        variable        THERMO_FREQ     equal 10
        variable        TEMP            equal 200.000000
        variable        PRES            equal 50000.000000
        variable        TAU_T           equal 0.100000
        variable        TAU_P           equal 0.500000
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
        pair_style      meam
        pair_coeff      * * library_18Metals.meam Sn Sn_18Metals.meam Sn
        # --------------------- MD SETTINGS ----------------------
        neighbor        1.0 bin
        timestep        0.002
        thermo          ${THERMO_FREQ}
        compute         allmsd all msd
        thermo_style    custom step ke pe etotal enthalpy temp press vol c_allmsd[*]
        # dump            1 all custom ${DUMP_FREQ} traj.dump id type x y z
        fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}
        fix             mzero all momentum 10 linear 1 1 1
        # --------------------- INITIALIZE -----------------------
        velocity        all create ${TEMP} 7858
        velocity        all zero linear
        # --------------------- RUN ------------------------------
        run             ${NSTEPS}
        write_data      out.lmp
        """)
        ret2 = ti._gen_lammps_input(**input)
        self.assertEqual(ret1, ret2)


if __name__ == '__main__':
    unittest.main()
