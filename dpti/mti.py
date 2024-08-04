#!/usr/bin/env python3

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from dpdispatcher import Machine, Resources, Submission, Task

from dpti.lib.lammps import get_natoms, get_thermo
from dpti.lib.utils import (
    block_avg,
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
    template_ff,
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
    ret += f"variable        TEMP            equal {temp:f}\n"
    ret += f"variable        PRES            equal {pres:f}\n"
    ret += f"variable        TAU_T           equal {tau_t:f}\n"
    ret += f"variable        TAU_P           equal {tau_p:f}\n"
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
    if model is not None:
        ret += f"pair_style      deepmd {model}\n"
        ret += "pair_coeff * *\n"
    elif template_ff is not None:
        ret += template_ff
    ret += "# --------------------- MD SETTINGS ----------------------\n"
    ret += "neighbor        2.0 bin\n"
    ret += "neigh_modify    every 10 delay 0 check no\n"
    ret += f"timestep        {timestep}\n"
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
        raise RuntimeError(f"unknow ensemble {ens}\n")
    if ens == "nvt" or ens == "nve":
        ret += "thermo_style    custom step temp f_1[5] f_1[7] c_allmsd[*]\n"
        ret += "thermo_modify   format float %6.8e\n"
        ret += 'fix print all print ${THERMO_FREQ} "$(step) $(temp) $(f_1[5]) $(f_1[7])" append ${ibead}.out title "# step temp K_prim K_cv" screen no'
    elif "npt" in ens:
        ret += "thermo_style    custom step temp vol density f_1[5] f_1[7] f_1[8] f_1[10] c_allmsd[*]\n"
        ret += "thermo_modify   format float %6.8e\n"
        ret += 'fix print all print ${THERMO_FREQ} "$(step) $(temp) $(vol) $(density) $(f_1[5]) $(f_1[7])" append ${ibead}.out title "# step temp vol density K_prim K_cv" screen no'
    else:
        raise RuntimeError(f"unknow ensemble {ens}\n")
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
    model = jdata.get("model", None)
    template_ff_file = jdata.get("template_ff", None)
    template_ff = None
    if template_ff_file is not None:
        with open(template_ff_file) as f:
            template_ff = f.read()
    if model is not None and template_ff is not None:
        raise RuntimeError(
            "You are providing both a dp model and a template forcefield. You can only set one of model and template_ff."
        )
    if model is None and template_ff is None:
        raise RuntimeError(
            "You must provide a dp model or a template forcefield. Please set either model or template_ff."
        )
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
    assert (
        job_type == "nbead_convergence" or job_type == "mass_ti"
    ), "Unknow job_type. Only nbead_convergence and mass_ti are supported."
    mass_scale_y_seq = get_first_matched_key_from_dict(jdata, ["mass_scale_y"])
    mass_scale_y_list = parse_seq(mass_scale_y_seq)
    mass_scales = (1.0 / np.array(mass_scale_y_list)) ** 2
    nbead_seq = get_first_matched_key_from_dict(jdata, ["nbead"])
    nbead_list = parse_seq(nbead_seq)
    nnode_seq = jdata.get("nnode", None)
    if nnode_seq is not None:
        nnode_list = parse_seq(nnode_seq)
        assert (
            len(nbead_list) == len(nnode_list)
        ), "Lists nbead and nnode should have same length. Please specify one nnode for each nbead."
    if job_type == "mass_ti":
        assert (
            len(mass_scale_y_list) == len(nbead_list)
        ), "For mass TI tasks, you must provide one value of nbead for each value of mass_scale_y."

    job_abs_dir = create_path(iter_name)
    ti_settings["equi_conf"] = relative_link_file(equi_conf, job_abs_dir)
    if model is not None:
        ti_settings["model"] = relative_link_file(model, job_abs_dir)
    if template_ff is not None:
        ti_settings["template_ff"] = relative_link_file(template_ff_file, job_abs_dir)
    with open(os.path.join(job_abs_dir, "mti_settings.json"), "w") as f:
        json.dump(ti_settings, f, indent=4)

    for ii in range(ntasks):
        task_dir = os.path.join(job_abs_dir, "task.%06d" % ii)
        task_abs_dir = create_path(task_dir)
        settings = {}
        if path == "t":
            temp = temp_list[ii]
        elif path == "p":
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
                    if nnode_seq is not None:
                        settings["nnode"] = nnode_list[kk]
                    relative_link_file(equi_conf, nbead_abs_dir)
                    task_model = model
                    if model is not None:
                        relative_link_file(model, nbead_abs_dir)
                        task_model = os.path.basename(model)
                        lmp_str = _gen_lammps_input(
                            os.path.basename(equi_conf),
                            mass_map,
                            mass_scales[jj],
                            task_model,
                            None,
                            nbead_list[kk],
                            nsteps,
                            timestep,
                            ens,
                            temp_list[ii],
                            pres=pres,
                            tau_t=tau_t,
                            thermo_freq=thermo_freq,
                            dump_freq=dump_freq,
                            copies=copies,
                        )
                    elif template_ff is not None:
                        lmp_str = _gen_lammps_input(
                            os.path.basename(equi_conf),
                            mass_map,
                            mass_scales[jj],
                            None,
                            template_ff,
                            nbead_list[kk],
                            nsteps,
                            timestep,
                            ens,
                            temp_list[ii],
                            pres=pres,
                            tau_t=tau_t,
                            thermo_freq=thermo_freq,
                            dump_freq=dump_freq,
                            copies=copies,
                        )
                    with open(os.path.join(nbead_abs_dir, "in.lammps"), "w") as f:
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
                if nnode_seq is not None:
                    settings["nnode"] = nnode_list[jj]
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
                        None,
                        nbead_list[jj],
                        nsteps,
                        timestep,
                        ens,
                        temp_list[ii],
                        pres=pres,
                        tau_t=tau_t,
                        thermo_freq=thermo_freq,
                        dump_freq=dump_freq,
                        copies=copies,
                    )
                elif template_ff is not None:
                    lmp_str = _gen_lammps_input(
                        os.path.basename(equi_conf),
                        mass_map,
                        mass_scales[jj],
                        None,
                        template_ff,
                        nbead_list[jj],
                        nsteps,
                        timestep,
                        ens,
                        temp_list[ii],
                        pres=pres,
                        tau_t=tau_t,
                        thermo_freq=thermo_freq,
                        dump_freq=dump_freq,
                        copies=copies,
                    )
                with open(os.path.join(mass_scale_y_abs_dir, "in.lammps"), "w") as f:
                    f.write(lmp_str)
                with open(
                    os.path.join(mass_scale_y_abs_dir, "settings.json"), "w"
                ) as f:
                    json.dump(settings, f, indent=4)


def run_task(task_name, jdata, machine_file):
    job_type = jdata["job_type"]
    nprocs_per_bead = jdata.get("nprocs_per_bead", 1)
    if job_type == "nbead_convergence":
        task_dir_list = glob.glob(
            os.path.join(task_name, "task.*/mass_scale_y.*/nbead.*")
        )
        link_model = "ln -s ../../../graph.pb"
    elif job_type == "mass_ti":
        task_dir_list = glob.glob(os.path.join(task_name, "task.*/mass_scale_y.*"))
        link_model = "ln -s ../../graph.pb"
    else:
        raise RuntimeError(
            "Unknow job_type. Only nbead_convergence and mass_ti are supported."
        )
    task_dir_list = sorted(task_dir_list)
    work_base_dir = os.getcwd()
    with open(machine_file) as f:
        mdata = json.load(f)
    task_exec = mdata["command"]
    number_node = mdata.get("resources", {}).get("number_node", 1)

    machine = Machine.load_from_dict(mdata["machine"])
    for ii in task_dir_list:
        setting = json.load(open(os.path.join(ii, "settings.json")))
        nbead = int(setting["nbead"])
        nnode = setting.get("nnode", None)
        if nnode is not None:
            mdata["resources"]["number_node"] = int(nnode)
            number_node = nnode
        mdata["resources"]["cpu_per_node"] = int(
            np.ceil(nbead * nprocs_per_bead / number_node)
        )
        resources = Resources.load_from_dict(mdata["resources"])

        submission = Submission(
            work_base=work_base_dir,
            resources=resources,
            machine=machine,
        )

        task = Task(
            command=f"{link_model}; if ls *.restart.100000 1> /dev/null 2>&1; then {task_exec} -in in.lammps -p {nbead}x{nprocs_per_bead} -log log -v restart 1; else {task_exec} -in in.lammps -p {nbead}x{nprocs_per_bead} -log log -v restart 0; fi",
            task_work_path=ii,
            forward_files=["in.lammps", "*.lmp", "graph.pb"],
            backward_files=["log*", "*out.lmp", "*.dump"],
        )

        submission.forward_common_files = []
        submission.register_task_list(task_list=[task])
        submission.run_submission(exit_on_submit=True)


def post_tasks(iter_name, jdata, natoms_mol=None):
    equi_conf = get_task_file_abspath(iter_name, jdata["equi_conf"])
    natoms = get_natoms(equi_conf)
    if "copies" in jdata:
        natoms *= np.prod(jdata["copies"])
    if natoms_mol is not None:
        natoms /= natoms_mol
    job_type = jdata["job_type"]
    stat_skip = jdata["stat_skip"]
    stat_bsize = jdata["stat_bsize"]
    if job_type == "nbead_convergence":
        counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        task_dir_list = glob.glob(
            os.path.join(iter_name, "task.*/mass_scale_y.*/nbead.*")
        )
        task_dir_list = sorted(task_dir_list)
        for ii in task_dir_list:
            parts = ii.split(os.sep)
            task, mass_scale_y, nbead = None, None, None
            for part in parts:
                if part.startswith("task."):
                    task = part
                elif part.startswith("mass_scale_y."):
                    mass_scale_y = part
                elif part.startswith("nbead."):
                    nbead = part
            counts[task][mass_scale_y][nbead] += 1

    elif job_type == "mass_ti":
        counts = defaultdict(lambda: defaultdict(int))
        task_dir_list = glob.glob(os.path.join(iter_name, "task.*/mass_scale_y.*"))
        task_dir_list = sorted(task_dir_list)
        for ii in task_dir_list:
            parts = ii.split(os.sep)
            task, mass_scale_y = None, None
            for part in parts:
                if part.startswith("task."):
                    task = part
                elif part.startswith("mass_scale_y."):
                    mass_scale_y = part
            counts[task][mass_scale_y] += 1
    else:
        raise RuntimeError(
            "Unknow job_type. Only nbead_convergence and mass_ti are supported."
        )
    ens = jdata["ens"]
    if ens == "nvt" or ens == "nve":
        stat_col = 3
    elif "npt" in ens:
        stat_col = 5
    else:
        raise RuntimeError("unsupported ens")

    if job_type == "nbead_convergence":
        for task in counts.keys():
            for mass_scale_y in counts[task].keys():
                result = []
                for nbead in counts[task][mass_scale_y].keys():
                    if counts[task][mass_scale_y][nbead] == 0:
                        continue
                    task_dir = os.path.join(iter_name, task, mass_scale_y, nbead)
                    settings = json.load(open(os.path.join(task_dir, "settings.json")))
                    out_files = glob.glob(os.path.join(task_dir, "*1.out"))
                    if len(out_files) == 0:
                        log_name = os.path.join(task_dir, "log.0")
                        data = get_thermo(log_name)
                        np.savetxt(os.path.join(task_dir, "data"), data, fmt="%.6e")
                    else:
                        out_files = sorted(out_files)
                        data = np.loadtxt(out_files[0])
                    num_nbead = settings["nbead"]
                    kcv, kcverr = block_avg(
                        data[:, stat_col], skip=stat_skip, block_size=stat_bsize
                    )
                    result.append([num_nbead, kcv, kcverr])
                result = np.array(result)
                np.savetxt(
                    os.path.join(iter_name, task, mass_scale_y, "kcv.out"),
                    result,
                    fmt=["%12d", "%22.6e", "%22.6e"],
                    header=f"{'nbead':>12} {'kcv':>22} {'kcv_err':>22}",
                )
    elif job_type == "mass_ti":
        for task in counts.keys():
            result = []
            for mass_scale_y in counts[task].keys():
                if counts[task][mass_scale_y] == 0:
                    continue
                task_dir = os.path.join(iter_name, task, mass_scale_y)
                settings = json.load(open(os.path.join(task_dir, "settings.json")))
                out_files = glob.glob(os.path.join(task_dir, "*1.out"))
                if len(out_files) == 0:
                    log_name = os.path.join(task_dir, "log.0")
                    data = get_thermo(log_name)
                    np.savetxt(os.path.join(task_dir, "data"), data, fmt="%.6e")
                else:
                    out_files = sorted(out_files)
                    data = np.loadtxt(out_files[0])
                mass_scale_y_value = settings["mass_scale_y"]
                kcv, kcverr = block_avg(
                    data[:, stat_col], skip=stat_skip, block_size=stat_bsize
                )
                result.append([mass_scale_y_value, kcv, kcverr])
            result = np.array(result)
            np.savetxt(
                os.path.join(iter_name, task, "kcv.out"),
                result,
                fmt=["%22.6e", "%22.6e", "%22.6e"],
                header="# mass_scale_y kcv kcv_err",
            )
            mass_scale_y_values, kcv_inte, kcv_inte_err, kcv_stat_err = integrate_range(
                result[:, 0], result[:, 1], result[:, 2]
            )
            np.savetxt(
                os.path.join(iter_name, task, "kcv_inte.out"),
                np.array([kcv_inte, kcv_inte_err, kcv_stat_err]),
                fmt=["%22.6e", "%22.6e", "%22.6e"],
                header="# kcv_inte kcv_inte_err kcv_stat_err",
            )
    else:
        raise RuntimeError(
            "Unknow job_type. Only nbead_convergence and mass_ti are supported."
        )


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
        "mti",
        help="mass thermodynamic integration: quantum free energy calculation using PIMD",
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

    parser_run = module_subparsers.add_parser("run", help="run the job")
    parser_run.add_argument("JOB", type=str, help="folder of the job")
    parser_run.add_argument("PARAM", type=str, help="json parameter file")
    parser_run.add_argument("machine", type=str, help="machine.json file for the job")
    parser_run.set_defaults(func=handle_run)

    parser_compute = module_subparsers.add_parser(
        "compute", help="Compute the result of a job"
    )
    parser_compute.add_argument("JOB", type=str, help="folder of the job")
    parser_compute.add_argument(
        "--natom_mol", type=int, help="the number of atoms in the molecule"
    )
    parser_compute.set_defaults(func=handle_compute)


def handle_gen(args):
    with open(args.PARAM) as j:
        jdata = json.load(j)
    make_tasks(args.output, jdata)


def handle_run(args):
    with open(args.PARAM) as j:
        jdata = json.load(j)
    run_task(args.JOB, jdata, args.machine)


def handle_compute(args):
    with open(os.path.join(args.JOB, "mti_settings.json")) as j:
        jdata = json.load(j)
    post_tasks(args.JOB, jdata, args.natom_mol)


if __name__ == "__main__":
    _main()
