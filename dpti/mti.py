#!/usr/bin/env python3

import glob
import json
import os
import shutil

import numpy as np
import pymbar
import scipy.constants as pc
from dpdispatcher import Machine, Resources, Submission, Task

from dpti.lib.lammps import get_natoms, get_thermo

from dpti.lib.utils import (
    block_avg,
    compute_nrefine,
    create_path,
    get_first_matched_key_from_dict,
    get_task_file_abspath,
    integrate_range,
    parse_seq,
    relative_link_file,
)

def _gen_lammps_input(
    conf_file,
    mass_map,
    mass_scale,
    model,
    nbeads,
    nsteps,
    timestep,
    ens,
    temp,
    pres=1.0,
    tau_t=0.1,
    tau_p=0.5,
    thermo_freq=100,
    dump_freq=100,
    copies=None,
    if_meam=False,
    meam_model=None,
):
    if nbeads is not None:
        if nbeads <= 0:
            raise ValueError(
                "The number of beads should be positive. Check your nbeads setting."
            )
        power = 1
        while power < nbeads:
            power *= 10
    ret = ""
    ret += "clear\n"
    ret += "# --------------------- VARIABLES-------------------------\n"
    ret += "variable        ibead           uloop %d pad\n" % (power - 1)
    ret += "variable        NSTEPS          equal %d\n" % nsteps
    ret += "variable        THERMO_FREQ     equal %d\n" % thermo_freq
    ret += "variable        DUMP_FREQ       equal %d\n" % dump_freq
    ret += "variable        TEMP            equal %f\n" % temp
    ret += "variable        PRES            equal %f\n" % pres
    ret += "variable        TAU_T           equal %f\n" % tau_t
    ret += "variable        TAU_P           equal %f\n" % tau_p
    ret += "# ---------------------- INITIALIZAITION ------------------\n"
    ret += "units           metal\n"
    ret += "boundary        p p p\n"
    ret += "atom_style      atomic\n"
    ret += "# --------------------- ATOM DEFINITION ------------------\n"
    ret += "box             tilt large\n"
    ret += f'if "${{restart}} > 0" then "read_restart ${{ibead}}.restart.*" else "read_data {conf_file}"\n'
    if copies is not None:
        ret += "replicate       %d %d %d\n" % (copies[0], copies[1], copies[2])
    for jj in range(len(mass_map)):
        ret += "mass            %d %f\n" % (jj + 1, mass_map[jj] * mass_scale)
    ret += "# --------------------- FORCE FIELDS ---------------------\n"
    if if_meam:
        ret += "pair_style      meam\n"
        ret += f'pair_coeff      * * {meam_model["library"]} {meam_model["element"]} {meam_model["potential"]} {meam_model["element"]}\n'
    else:
        ret += "pair_style      deepmd %s\n" % model
        ret += "pair_coeff * *\n"
    ret += "# --------------------- MD SETTINGS ----------------------\n"
    ret += "neighbor        2.0 bin\n"
    ret += "neigh_modify    every 10 delay 0 check yes\n"
    ret += "timestep        %s\n" % timestep
    ret += "thermo          ${THERMO_FREQ}\n"
    ret += "compute         allmsd all msd\n"
    ret += "dump            1 all custom ${DUMP_FREQ} ${ibead}.dump id type x y z\n"
    if ens == "nvt":
        ret += "fix 1 all pimd/langevin ensemble nvt integrator obabo temp ${TEMP} thermostat PILE_L 1234 tau ${TAU_T}\n"
    elif ens == "npt-iso" or ens == "npt":
        ret += "fix 1 all pimd/langevin ensemble npt integrator obabo temp ${TEMP} thermostat PILE_L 1234 tau ${TAU_T} iso ${PRES} barostat BZP taup ${TAU_P}\n"
    elif ens == "npt-aniso":
        ret += "fix 1 all pimd/langevin ensemble npt integrator obabo temp ${TEMP} thermostat PILE_L 1234 tau ${TAU_T} aniso ${PRES} barostat BZP taup ${TAU_P}\n"
    elif ens == "npt-tri":
        raise RuntimeError("npt-tri is not supported yet")
    elif ens == "npt-xy":
        raise RuntimeError("npt-xy is not supported yet")
    elif ens == "nve":
        ret += "fix 1 all pimd/langevin ensemble nve integrator obabo temp ${TEMP}\n"
    else:
        raise RuntimeError("unknow ensemble %s\n" % ens)
    if ens == "nvt" or ens == "nve":
        ret += "thermo_style    custom step temp f_1[5] f_1[7] c_allmsd[*]\n"
        ret += "thermo_modify   format 3*4 %6.8e\n"
    elif "npt" in ens:
        ret += "thermo_style    custom step temp vol density f_1[5] f_1[7] f_1[8] f_1[10] c_allmsd[*]\n"
        ret += "thermo_modify   format 3*4 %6.8e\n"
    else:
        raise RuntimeError("unknow ensemble %s\n" % ens)
    ret += "# --------------------- INITIALIZE -----------------------\n"
    ret += "# --------------------- RUN ------------------------------\n"
    ret += "restart         100000 ${ibead}.restart\n"
    ret += "run             ${NSTEPS} upto\n"
    ret += "write_data      ${ibead}.out.lmp\n"

    return ret

def make_tasks(iter_name, jdata, if_meam=None):
    ti_settings = jdata.copy()
    if if_meam is None:
        if_meam = jdata.get("if_meam", None)
    equi_conf = jdata["equi_conf"]
    equi_conf = os.path.abspath(equi_conf)
    copies = None
    if "copies" in jdata:
        copies = jdata["copies"]
    model = jdata["model"]
    meam_model = jdata.get("meam_model", None)
    mass_map = get_first_matched_key_from_dict(jdata, ["model_mass_map", "mass_map"])
    nsteps = jdata["nsteps"]
    timestep = get_first_matched_key_from_dict(jdata, ["timestep", "dt"])
    thermo_freq = get_first_matched_key_from_dict(jdata, ["thermo_freq", "stat_freq"])
    dump_freq = get_first_matched_key_from_dict(
        jdata, ["dump_freq", "thermo_freq", "stat_freq"]
    )
    ens = jdata["ens"]
    path = jdata["path"]
    mass_scale_y_seq = get_first_matched_key_from_dict(jdata, ["mass_scale_y_seq"])
    mass_scale_y_list = parse_seq(mass_scale_y_seq)
    ntasks = len(mass_scale_y_list)
    if "nvt" in ens:
        if path == "t":
            
            temp_list = parse_seq(temp_seq)
            tau_t = jdata["tau_t"]
            ntasks = len(temp_list)
        else:
            raise RuntimeError("supported path of nvt ens is 't'")
    elif "npt" in ens:
        if path == "t":
            temp_seq = get_first_matched_key_from_dict(jdata, ["temp_seq", "temps"])
            temp_list = parse_seq(temp_seq)
            pres = get_first_matched_key_from_dict(jdata, ["pres", "press"])
            ntasks = len(temp_list)
        elif path == "t-ginv":
            temp_seq = get_first_matched_key_from_dict(jdata, ["temp_seq", "temps"])
            temp_list = parse_seq_ginv(temp_seq)
            pres = get_first_matched_key_from_dict(jdata, ["pres", "press"])
            ntasks = len(temp_list)
        elif path == "p":
            temp = get_first_matched_key_from_dict(jdata, ["temp", "temps"])
            pres_seq = get_first_matched_key_from_dict(jdata, ["pres_seq", "press"])
            pres_list = parse_seq(pres_seq)
            ntasks = len(pres_list)
        else:
            raise RuntimeError("supported path of npt ens are 't' or 'p'")
        tau_t = jdata["tau_t"]
        tau_p = jdata["tau_p"]
    else:
        raise RuntimeError("invalid ens")

    job_abs_dir = create_path(iter_name)