import os, textwrap
import numpy as np
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from context import dpti
from potential_common import meam_model

class TestEquiGenLammpsInput(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    @patch('numpy.random')
    def test_deepmd_npt(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        input = dict(nsteps=1000000, 
            thermo_freq=10, 
            dump_freq=100000, 
            mass_map=[118.71],
            temp=400, 
            tau_t=0.2,
            tau_p=2.0, 
            equi_conf='conf.lmp',
            model='graph.pb',
            timestep=0.002, 
            if_dump_avg_posi=False,
            ens='npt-iso',
            pres=200000,
            if_meam=False,
            meam_model=None,
        )

        ret1 = textwrap.dedent("""\
        clear
        # --------------------- VARIABLES-------------------------
        variable        NSTEPS          equal 1000000
        variable        THERMO_FREQ     equal 10
        variable        DUMP_FREQ       equal 100000
        variable        NREPEAT         equal ${NSTEPS}/${DUMP_FREQ}
        variable        TEMP            equal 400.000000
        variable        PRES            equal 200000.000000
        variable        TAU_T           equal 0.200000
        variable        TAU_P           equal 2.000000
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
        timestep        0.002000
        thermo          ${THERMO_FREQ}
        compute         allmsd all msd
        thermo_style    custom step ke pe etotal enthalpy temp press vol lx ly lz xy xz yz pxx pyy pzz pxy pxz pyz c_allmsd[*]
        dump            1 all custom ${DUMP_FREQ} dump.equi id type x y z vx vy vz
        fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}
        fix             mzero all momentum 10 linear 1 1 1
        # --------------------- INITIALIZE -----------------------
        velocity        all create ${TEMP} 7858
        velocity        all zero linear
        # --------------------- RUN ------------------------------
        run             ${NSTEPS}
        write_data      out.lmp
        """)

        ret2 = dpti.equi.gen_equi_lammps_input(**input)
        self.assertEqual(ret1, ret2)

    @patch('numpy.random')
    def test_deepmd_nvt(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        input = dict(nsteps=1000000, 
            thermo_freq=10, 
            dump_freq=100000, 
            mass_map=[118.71],
            temp=400, 
            tau_t=0.2,
            tau_p=2.0, 
            equi_conf='conf.lmp',
            model='graph.pb',
            timestep=0.002, 
            if_dump_avg_posi=False,
            ens='nvt',
            pres=200000,
            if_meam=False,
            meam_model=None,
        )
        ret2 = dpti.equi.gen_equi_lammps_input(**input)

        ret1 = textwrap.dedent("""\
        clear
        # --------------------- VARIABLES-------------------------
        variable        NSTEPS          equal 1000000
        variable        THERMO_FREQ     equal 10
        variable        DUMP_FREQ       equal 100000
        variable        NREPEAT         equal ${NSTEPS}/${DUMP_FREQ}
        variable        TEMP            equal 400.000000
        variable        PRES            equal 200000.000000
        variable        TAU_T           equal 0.200000
        variable        TAU_P           equal 2.000000
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
        timestep        0.002000
        thermo          ${THERMO_FREQ}
        compute         allmsd all msd
        thermo_style    custom step ke pe etotal enthalpy temp press vol lx ly lz xy xz yz pxx pyy pzz pxy pxz pyz c_allmsd[*]
        dump            1 all custom ${DUMP_FREQ} dump.equi id type x y z vx vy vz
        fix             1 all nvt temp ${TEMP} ${TEMP} ${TAU_T}
        fix             mzero all momentum 10 linear 1 1 1
        # --------------------- INITIALIZE -----------------------
        velocity        all create ${TEMP} 7858
        velocity        all zero linear
        # --------------------- RUN ------------------------------
        run             ${NSTEPS}
        write_data      out.lmp
        """)
        self.assertEqual(ret1, ret2)

    @patch('numpy.random')
    def test_meam_npt(self, patch_random):
        patch_random.randint = MagicMock(return_value=7858)
        input = dict(nsteps=1000000, 
            thermo_freq=10, 
            dump_freq=100000, 
            mass_map=[118.71],
            temp=400, 
            tau_t=0.2,
            tau_p=2.0, 
            equi_conf='conf.lmp',
            model=None,
            timestep=0.002, 
            if_dump_avg_posi=False,
            ens='npt-xy',
            pres=200000,
            if_meam=True,
            meam_model=meam_model,
        )
        ret2 = dpti.equi.gen_equi_lammps_input(**input)
        ret1 = textwrap.dedent("""\
        clear
        # --------------------- VARIABLES-------------------------
        variable        NSTEPS          equal 1000000
        variable        THERMO_FREQ     equal 10
        variable        DUMP_FREQ       equal 100000
        variable        NREPEAT         equal ${NSTEPS}/${DUMP_FREQ}
        variable        TEMP            equal 400.000000
        variable        PRES            equal 200000.000000
        variable        TAU_T           equal 0.200000
        variable        TAU_P           equal 2.000000
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
        timestep        0.002000
        thermo          ${THERMO_FREQ}
        compute         allmsd all msd
        thermo_style    custom step ke pe etotal enthalpy temp press vol lx ly lz xy xz yz pxx pyy pzz pxy pxz pyz c_allmsd[*]
        dump            1 all custom ${DUMP_FREQ} dump.equi id type x y z vx vy vz
        fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} aniso ${PRES} ${PRES} ${TAU_P} couple xy
        fix             mzero all momentum 10 linear 1 1 1
        # --------------------- INITIALIZE -----------------------
        velocity        all create ${TEMP} 7858
        velocity        all zero linear
        # --------------------- RUN ------------------------------
        run             ${NSTEPS}
        write_data      out.lmp
        """)
        self.assertEqual(ret1, ret2)


if __name__ == '__main__':

    unittest.main()
