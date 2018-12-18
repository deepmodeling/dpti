#!/usr/bin/env python3

import os, sys, json, argparse, glob, shutil
import numpy as np
import scipy.constants as pc

from lib.utils import create_path
from lib.utils import copy_file_list
from lib.utils import block_avg
from lib.utils import integrate
from lib.utils import integrate_sys_err
from lib.utils import parse_seq
from lib.lammps import get_thermo
from lib.lammps import get_natoms

import ti

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
    parser_comp.add_argument('-e', '--Eo', type=float, default = 0,
                             help='free energy of starting point')
    args = parser.parse_args()

    if args.command is None :
        parser.print_help()
        exit
    if args.command == 'gen' :
        output = args.output
        jdata = json.load(open(args.PARAM, 'r'))
        ti.make_tasks(output, jdata)
    elif args.command == 'compute' :
        job = args.JOB
        jdata = json.load(open(os.path.join(job, 'in.json'), 'r'))
        equi_conf = jdata['equi_conf']
        natoms = get_natoms(equi_conf)
        if 'copies' in jdata :
            natoms *= np.prod(jdata['copies'])
        nmols = natoms // 3
        e0 = float(args.Eo)
        ti.post_tasks(job, jdata, e0, natoms = nmols)

    
if __name__ == '__main__' :
    _main()
        