#!/usr/bin/env python3

import argparse
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
    copies=None
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
    ret += "atom_modify     map yes\n"
    ret += "# --------------------- ATOM DEFINITION ------------------\n"
    ret += "box             tilt large\n"
    ret += f'if "${{restart}} > 0" then "read_restart ${{ibead}}.restart.*" else "read_data {conf_file}"\n'
    if copies is not None:
        ret += "replicate       %d %d %d\n" % (copies[0], copies[1], copies[2])
    for jj in range(len(mass_map)):
        ret += "mass            %d %f\n" % (jj + 1, mass_map[jj] * mass_scale)
    ret += "# --------------------- FORCE FIELDS ---------------------\n"
    ret += "pair_style      deepmd %s\n" % model
    ret += "pair_coeff * *\n"
    ret += "# --------------------- MD SETTINGS ----------------------\n"
    ret += "neighbor        2.0 bin\n"
    ret += "neigh_modify    every 10 delay 0 check no\n"
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

def make_tasks(iter_name, jdata):
    ti_settings = jdata.copy()
    equi_conf = jdata["equi_conf"]
    equi_conf = os.path.abspath(equi_conf)
    copies = None
    if "copies" in jdata:
        copies = jdata["copies"]
    model = jdata["model"]
    mass_map = get_first_matched_key_from_dict(jdata, ["model_mass_map", "mass_map"])
    nsteps = jdata["nsteps"]
    timestep = get_first_matched_key_from_dict(jdata, ["timestep", "dt"])
    thermo_freq = get_first_matched_key_from_dict(jdata, ["thermo_freq", "stat_freq"])
    dump_freq = get_first_matched_key_from_dict(
        jdata, ["dump_freq", "thermo_freq", "stat_freq"]
    )
    ens = jdata["ens"]
    path = jdata["path"]
    if "npt" in ens:
        if path == "t":
            temp_seq = get_first_matched_key_from_dict(jdata, ["temp_seq", "temps"])
            temp_list = parse_seq(temp_seq)
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
    
    job_type = jdata["job_type"]
    assert job_type == "nbead_convergence" or job_type == "mass_ti", "Unknow job_type. Only nbead_convergence and mass_ti are supported."
    mass_scale_y_seq = get_first_matched_key_from_dict(jdata, ["mass_scale_y"])
    mass_scale_y_list = parse_seq(mass_scale_y_seq)
    mass_scales = (1./np.array(mass_scale_y_list))**2
    nbead_seq = get_first_matched_key_from_dict(jdata, ["nbead"])
    nbead_list = parse_seq(nbead_seq)
    if job_type == "mass_ti":
        assert len(mass_scale_y_list) == len(nbead_list), "For mass TI tasks, you must provide one value of nbead for each value of mass_scale_y."

    job_abs_dir = create_path(iter_name)
    ti_settings["equi_conf"] = relative_link_file(equi_conf, job_abs_dir)
    ti_settings["model"] = relative_link_file(model, job_abs_dir)
    with open(os.path.join(job_abs_dir, "mti_settings.json"), "w") as f:
        json.dump(ti_settings, f, indent=4)

    for ii in range(ntasks):
        task_dir = os.path.join(job_abs_dir, "task.%06d" % ii)
        task_abs_dir = create_path(task_dir)
        settings = {}
        if path == "t":
            temp = temp_list[ii]
            pres = pres
        elif path == "p":
            temp = temp
            pres = pres_list[ii]
        else:
            raise RuntimeError("unsupported path")
        settings["temp"] = temp
        settings["pres"] = pres
        if job_type == "nbead_convergence":
            for jj in range(len(mass_scale_y_list)):
                mass_scale_y_dir = os.path.join(task_abs_dir, "mass_scale_y.%06d" % jj)
                mass_scale_y_abs_dir = create_path(mass_scale_y_dir)
                settings["mass_scale_y"] = mass_scale_y_list[jj]
                settings["mass_scale"] = mass_scales[jj]
                for kk in range(len(nbead_list)):
                    nbead_dir = os.path.join(mass_scale_y_abs_dir, "nbead.%06d" % kk)
                    nbead_abs_dir = create_path(nbead_dir)
                    settings["nbead"] = nbead_list[kk]
                    relative_link_file(equi_conf, nbead_abs_dir)
                    task_model = model
                    if model:
                        relative_link_file(model, nbead_abs_dir)
                        task_model = os.path.basename(model)
                    lmp_str = _gen_lammps_input(
                        os.path.basename(equi_conf),
                        mass_map,
                        mass_scales[jj],
                        task_model,
                        nbead_list[kk],
                        nsteps,
                        timestep,
                        ens,
                        temp_list[ii],
                        pres=pres,
                        tau_t=tau_t,
                        thermo_freq=thermo_freq,
                        dump_freq=dump_freq,
                        copies=copies
                    )
                    with open(os.path.join(nbead_abs_dir, "in.lmp"), "w") as f:
                        f.write(lmp_str)
                    with open(os.path.join(nbead_abs_dir, "settings.json"), "w") as f:
                        json.dump(settings, f, indent=4)
        elif job_type == "mass_ti":
            for jj in range(len(mass_scale_y_list)):
                mass_scale_y_dir = os.path.join(task_abs_dir, "mass_scale_y.%06d" % jj)
                mass_scale_y_abs_dir = create_path(mass_scale_y_dir)
                settings["mass_scale_y"] = mass_scale_y_list[jj]
                settings["mass_scale"] = mass_scales[jj]
                settings["nbead"] = nbead_list[jj]
                relative_link_file(equi_conf, mass_scale_y_abs_dir)
                task_model = model
                if model:
                    relative_link_file(model, mass_scale_y_abs_dir)
                    task_model = os.path.basename(model)
                lmp_str = _gen_lammps_input(
                    os.path.basename(equi_conf),
                    mass_map,
                    mass_scales[jj],
                    task_model,
                    nbead_list[jj],
                    nsteps,
                    timestep,
                    ens,
                    temp_list[ii],
                    pres=pres,
                    tau_t=tau_t,
                    thermo_freq=thermo_freq,
                    dump_freq=dump_freq,
                    copies=copies
                )
                with open(os.path.join(mass_scale_y_abs_dir, "in.lmp"), "w") as f:
                    f.write(lmp_str)
                with open(os.path.join(mass_scale_y_abs_dir, "settings.json"), "w") as f:
                    json.dump(settings, f, indent=4)




def _main():
    parser = argparse.ArgumentParser(
        description="Compute free energy of ice by Hamiltonian TI"
    )
    main_subparsers = parser.add_subparsers(
        title="modules",
        description="the subcommands of dpti",
        help="module-level help",
        dest="module",
        required=True,
    )
    add_subparsers(main_subparsers)
    args = parser.parse_args()
    exec_args(args, parser)


def exec_args(args, parser):
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

def add_module_subparsers(main_subparsers):
    module_parser = main_subparsers.add_parser(
        "mti", help="mass thermodynamic integration: quantum free energy calculation using PIMD"
    )
    module_subparsers = module_parser.add_subparsers(
        help="commands of mass thermodynamic integration",
        dest="command",
        required=True,
    )
    add_subparsers(module_subparsers)

def add_subparsers(module_subparsers):
    parser_gen = module_subparsers.add_parser("gen", help="Generate a job")
    parser_gen.add_argument("PARAM", type=str, help="json parameter file")
    parser_gen.add_argument(
        "-o",
        "--output",
        type=str,
        default="new_job",
        help="the output folder for the job",
    )
    parser_gen.set_defaults(func=handle_gen)

def handle_gen(args):
    with open(args.PARAM) as j:
        jdata = json.load(j)
    make_tasks(args.output, jdata)

if __name__ == "__main__":
    _main()