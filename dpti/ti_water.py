#!/usr/bin/env python3

import os, sys, json, argparse, glob, shutil
import numpy as np
import scipy.constants as pc

from dpti.lib.utils import create_path
from dpti.lib.utils import copy_file_list
from dpti.lib.utils import block_avg
from dpti.lib.utils import integrate
from dpti.lib.utils import integrate_sys_err
from dpti.lib.utils import parse_seq
from dpti.lib.utils import get_task_file_abspath
from dpti.lib.lammps import get_thermo
from dpti.lib.lammps import get_natoms

from dpti import ti

def _main ():
    parser = argparse.ArgumentParser(
        description="Compute free energy by TI")
    subparsers = parser.add_subparsers(title='Valid subcommands', dest='command', help = 'valid commands')

    parser_gen = subparsers.add_parser('gen', help='Generate a job')
    parser_gen.add_argument('PARAM', type=str ,
                            help='json parameter file')
    parser_gen.add_argument('-o','--output', type=str, default = 'new_job',
                            help='the output folder for the job')

    parser_comp = subparsers.add_parser('compute', help= 'Compute the result of a job')
    parser_comp.add_argument('JOB', type=str ,
                             help='folder of the job')
    parser_comp.add_argument('-m','--inte-method', type=str, default = 'inte', 
                             choices=['inte', 'mbar'], 
                             help='the method of thermodynamic integration')
    parser_comp.add_argument('-e', '--Eo', type=float, default = 0,
                             help='free energy of starting point')
    parser_comp.add_argument('-E', '--Eo-err', type=float, default = 0,
                             help='The statistical error of the starting free energy')
    parser_comp.add_argument('-t', '--To', type=float, 
                             help='the starting thermodynamic position')
    parser_comp.add_argument('-s', '--scheme', type=str, default = 'simpson',
                             help='the numerical integration scheme')
    parser_comp.add_argument('-S', '--shift', type=float, default = 0.0,
                             help='a constant shift in the energy/mole computation, will be removed from FE')

    parser_comp = subparsers.add_parser('refine', help= 'Refine the grid of a job')
    parser_comp.add_argument('-i', '--input', type=str, required=True,
                             help='input job')
    parser_comp.add_argument('-o', '--output', type=str, required=True,
                             help='output job')
    parser_comp.add_argument('-e', '--error', type=float, required=True,
                             help='the error required')

    args = parser.parse_args()
    return exec_args(args=args, parser=parser)

def exec_args(args, parser=None):
    if args.command is None :
        parser.print_help()
        exit
    if args.command == 'gen' :
        output = args.output
        jdata = json.load(open(args.PARAM, 'r'))
        ti.make_tasks(output, jdata)
    elif args.command == 'compute' :
        job = args.JOB
        jdata = json.load(open(os.path.join(job, 'ti_settings.json'), 'r'))
        equi_conf = get_task_file_abspath(job, jdata['equi_conf'])
        natoms = get_natoms(equi_conf)
        if 'copies' in jdata :
            natoms *= np.prod(jdata['copies'])
        nmols = natoms // 3
        if args.inte_method == 'inte' :
            ti.post_tasks(job, jdata, args.Eo, Eo_err = args.Eo_err, To = args.To, natoms = nmols, scheme = args.scheme, shift = args.shift)
        elif args.inte_method == 'mbar' :
            ti.post_tasks_mbar(job, jdata, args.Eo, natoms = nmols)
        else :
            raise RuntimeError('unknow integration method')
    elif args.command == 'refine' :
        ti.refine_task(args.input, args.output, args.error)

if __name__ == '__main__' :
    _main()
        
