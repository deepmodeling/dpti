#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np

from dpti import ti
from dpti.lib.lammps import get_natoms
from dpti.lib.utils import get_task_file_abspath


def _main():
    parser = argparse.ArgumentParser(
        description="thermodynamic integration along isothermal or isobaric paths for water"
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
        "ti_water",
        help="thermodynamic integration along isothermal or isobaric paths for water",
    )
    module_subparsers = module_parser.add_subparsers(
        help="commands of thermodynamic integration along isothermal or isobaric paths for water",
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

    parser_compute = module_subparsers.add_parser(
        "compute", help="Compute the result of a job"
    )
    parser_compute.add_argument("JOB", type=str, help="folder of the job")
    parser_compute.add_argument(
        "-m",
        "--inte-method",
        type=str,
        default="inte",
        choices=["inte", "mbar"],
        help="the method of thermodynamic integration",
    )
    parser_compute.add_argument(
        "-e", "--Eo", type=float, default=None, help="free energy of starting point"
    )
    parser_compute.add_argument(
        "-E",
        "--Eo-err",
        type=float,
        default=None,
        help="the statistical error of the starting free energy",
    )
    parser_compute.add_argument(
        "-t", "--To", type=float, help="the starting thermodynamic position"
    )
    parser_compute.add_argument(
        "-s",
        "--scheme",
        type=str,
        default="simpson",
        help="the numerical integration scheme",
    )
    parser_compute.add_argument(
        "-S",
        "--shift",
        type=float,
        default=0.0,
        help="a constant shift in the energy/mole computation, will be removed from FE",
    )
    parser_compute.add_argument(
        "-H",
        "--hti",
        type=str,
        default=None,
        help="the HTI job folder; will extract the free energy of the starting point as from the result.json file in this folder",
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
    parser_refine.set_defaults(func=handle_refine)

    parser_run = module_subparsers.add_parser("run", help="run the job")
    parser_run.add_argument("JOB", type=str, help="folder of the job")
    parser_run.add_argument("machine", type=str, help="machine.json file for the job")
    parser_run.set_defaults(func=handle_run)


def handle_gen(args):
    output = args.output
    with open(args.PARAM) as j:
        jdata = json.load(j)
    ti.make_tasks(output, jdata)


def handle_compute(args):
    job = args.JOB
    jdata = json.load(open(os.path.join(job, "ti_settings.json")))
    path = jdata["path"]
    equi_conf = get_task_file_abspath(job, jdata["equi_conf"])
    natoms = get_natoms(equi_conf)
    if "copies" in jdata:
        natoms *= np.prod(jdata["copies"])
    nmols = natoms // 3
    hti_dir = args.hti
    jdata_hti = json.load(open(os.path.join(hti_dir, "result.json")))
    jdata_hti_in = json.load(open(os.path.join(hti_dir, "in.json")))
    if args.Eo is not None and args.hti is not None:
        raise Warning(
            "Both Eo and hti are provided. Eo will be overrided by the e1 value in hti's result.json file. Make sure this is what you want."
        )
    if args.Eo is None:
        args.Eo = jdata_hti["e1"]
    if args.Eo_err is None:
        args.Eo_err = jdata_hti["e1_err"]
    if args.To is None:
        if path == "t" or path == "t-ginv":
            args.To = jdata_hti_in["temp"]
        elif path == "p":
            args.To = jdata_hti_in["pres"]
    if args.inte_method == "inte":
        ti.post_tasks(
            job,
            jdata,
            args.Eo,
            Eo_err=args.Eo_err,
            To=args.To,
            natoms=nmols,
            scheme=args.scheme,
            shift=args.shift,
        )
    elif args.inte_method == "mbar":
        ti.post_tasks_mbar(job, jdata, args.Eo, natoms=nmols)
    else:
        raise RuntimeError("unknow integration method")


def handle_refine(args):
    ti.refine_task(args.input, args.output, args.error)


def handle_run(args):
    ti.run_task(args.JOB, args.machine)


if __name__ == "__main__":
    _main()
