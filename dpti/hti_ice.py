#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np
import scipy.constants as pc

from dpti import einstein, hti
from dpti.lib import lmp


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
        "hti_ice", help="Hamiltonian thermodynamic integration for ice"
    )
    module_subparsers = module_parser.add_subparsers(
        help="commands of Hamiltonian thermodynamic integration for ice",
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
    parser_gen.add_argument(
        "-s",
        "--switch",
        type=str,
        default="one-step",
        choices=["one-step", "two-step", "three-step"],
        help="one-step: switching on DP and switching off spring simultanenously.\
                            two-step: 1 switching on DP, 2 switching off spring.\n\
                            three-step: 1 switching on soft LJ, 2 switching on DP, 3 switching off spring and soft LJ.",
    )
    parser_gen.set_defaults(func=handle_gen)

    parser_compute = module_subparsers.add_parser(
        "compute", help="Compute the result of a job"
    )
    parser_compute.add_argument("JOB", type=str, help="folder of the job")
    parser_compute.add_argument(
        "-t",
        "--type",
        type=str,
        default="helmholtz",
        choices=["helmholtz", "gibbs"],
        help="the type of free energy",
    )
    parser_compute.add_argument(
        "-m",
        "--inte-method",
        type=str,
        default="inte",
        choices=["inte", "mbar"],
        help="the method of thermodynamic integration",
    )
    parser_compute.add_argument(
        "-d",
        "--disorder-corr",
        action="store_false",
        default=True,
        help="apply disorder correction for ice",
    )
    parser_compute.add_argument(
        "-p",
        "--partial-disorder",
        type=str,
        choices=["3", "5"],
        help="apply partial disorder correction for ice",
    )
    parser_compute.add_argument(
        "-s",
        "--scheme",
        type=str,
        default="simpson",
        help="the numeric integration scheme",
    )
    parser_compute.add_argument(
        "-S",
        "--shift",
        type=float,
        default=0.0,
        help="a constant shift in the energy/mole computation, will be removed from FE",
    )
    parser_compute.add_argument(
        "-g",
        "--pv",
        type=float,
        default=None,
        help="press*vol value override to calculate Gibbs free energy",
    )
    parser_compute.add_argument(
        "-G", "--pv-err", type=float, default=None, help="press*vol error"
    )
    parser_compute.add_argument(
        "--npt",
        type=str,
        default=None,
        help="directory of the npt task; will use PV from npt result, where P is the control variable and V varies.",
    )
    parser_compute.set_defaults(func=handle_compute)

    parser_refine = module_subparsers.add_parser(
        "refine", help="Refine the grid of a job"
    )
    parser_refine.add_argument(
        "-i", "--input", type=str, required=True, help="input job"
    )
    parser_refine.add_argument(
        "-o", "--output", type=str, required=True, help="output job"
    )
    parser_refine.add_argument(
        "-e", "--error", type=float, required=True, help="the error required"
    )
    parser_refine.add_argument(
        "-p", "--print", action="store_true", help="print the refinement and exit"
    )
    parser_refine.set_defaults(func=handle_refine)

    parser_run = module_subparsers.add_parser("run", help="run the job")
    parser_run.add_argument("JOB", type=str, help="folder of the job")
    parser_run.add_argument("machine", type=str, help="machine.json file for the job")
    parser_run.add_argument("task_name", type=str, help="task name, can be 00, 01, or 02")
    parser_run.add_argument("--use-dp", type=bool, default=True, help="whether to use Deep Potential or not")
    parser_run.set_defaults(func=handle_run)


def handle_gen(args):
    output = args.output
    with open(args.PARAM) as j:
        jdata = json.load(j)
    if "crystal" in jdata and jdata["crystal"] == "frenkel":
        print("# gen task with Frenkel's Einstein crystal")
    else:
        print("# gen task with Vega's Einstein molecule")
    hti.make_tasks(output, jdata, "einstein", args.switch)


def handle_refine(args):
    hti.refine_task(args.input, args.output, args.error, args.print)


def handle_compute(args):
    job = args.JOB
    jdata = json.load(open(os.path.join(job, "in.json")))
    fp_conf = open(os.path.join(args.JOB, "conf.lmp"))
    sys_data = lmp.to_system_data(fp_conf.read().split("\n"))
    natoms = sum(sys_data["atom_numbs"])
    if "copies" in jdata:
        natoms *= np.prod(jdata["copies"])
    nmols = natoms // 3
    # compute e0
    if "crystal" not in jdata:
        jdata["crystal"] = "vega"
    crystal = jdata["crystal"]
    if crystal == "vega":
        e0 = einstein.free_energy(job) * 3
    else:
        e0 = einstein.frenkel(job) * 3
    # compute Paulin estimate for disordered entropy
    if args.disorder_corr:
        temp = jdata["temp"]
        if args.partial_disorder is not None:
            if args.partial_disorder == "5":
                pauling_corr = -pc.Boltzmann * temp / pc.electron_volt * 0.3817
                note_pauling = "(ice5)"
            elif args.partial_disorder == "3":
                pauling_corr = -pc.Boltzmann * temp / pc.electron_volt * 0.3686
                note_pauling = "(ice3)"
            else:
                raise RuntimeError(f"unknow partial_disorder {args.partial_disorder}")
        else:
            pauling_corr = -pc.Boltzmann * temp / pc.electron_volt * np.log(1.5)
            note_pauling = "      "
        e0 += pauling_corr
    else:
        note_pauling = "      "
        pauling_corr = 0
    # compute integration
    de, de_err, thermo_info = hti.post_tasks(
        job, jdata, natoms=nmols, method=args.inte_method, scheme=args.scheme
    )
    info = thermo_info.copy()
    # printing
    print_format = "%20.12f  %10.3e  %10.3e"
    hti.print_thermo_info(thermo_info)
    if crystal == "vega":
        print("# free ener of Einstein Mole: %20.8f" % (e0))
    else:
        print("# free ener of Einstein Crys: %20.8f" % (e0))
    print(f"# Pauling corr {note_pauling}:        {pauling_corr:20.8f}")
    print(
        ("# fe integration              " + print_format) % (de, de_err[0], de_err[1])
    )
    print("# fe const shift              %20.12f" % args.shift)
    # if args.type == 'helmholtz' :
    print("# Helmholtz free ener per mol (stat_err inte_err) [eV]:")
    print(print_format % (e0 + de - args.shift, de_err[0], de_err[1]))
    if args.type == "helmholtz":
        e1 = e0 + de - args.shift
        e1_err = de_err[0]
    elif args.type == "gibbs":
        if args.npt is not None:
            npt_in = json.load(open(os.path.join(args.npt, "jdata.json")))
            npt_info = json.load(open(os.path.join(args.npt, "result.json")))
            p = npt_in["pres"]
            v = npt_info["v"]
            v_err = npt_info["v_err"]
            unit_cvt = 1e5 * (1e-10**3) / pc.electron_volt
            pv = p * v * unit_cvt * 3
            pv_err = p * v_err * unit_cvt * np.sqrt(3)
            print(f"# use pv from npt task: pv = {pv:.6e} pv_err = {pv_err:.6e}")
        if args.npt is None and args.pv is not None:
            pv = args.pv
            print(f"# use manual pv=={pv}")
        elif args.npt is None and args.pv is None:
            pv = thermo_info["pv"]
        if args.npt is None and args.pv_err is not None:
            pv_err = args.pv_err
            print(f"# use manual pv_err=={pv_err}")
        elif args.npt is None and args.pv_err is None:
            pv_err = thermo_info["pv_err"]
        e1 = e0 + de + pv - args.shift
        e1_err = np.sqrt(de_err[0] ** 2 + pv_err**2)
        print("# Gibbs free ener per mol (stat_err inte_err) [eV]:")
        print(print_format % (e1, e1_err, de_err[1]))
        info["pv"] = pv
        info["pv_err"] = pv_err
    else:
        raise RuntimeError("unknown free energy type")
    free_energy_type = args.type
    info["free_energy_type"] = free_energy_type
    # info['de'] = de
    # info['de_err'] = de_err
    info["e1"] = e1
    info["e1_err"] = e1_err
    with open(os.path.join(job, "result.json"), "w") as result:
        result.write(json.dumps(info))
    return info

def handle_run(args):
    hti.run_task(args.JOB, args.machine, args.task_name, args.use_dp)

if __name__ == "__main__":
    _main()
