#!/usr/bin/env python3

import os, sys, json, argparse, glob
import numpy as np

from lib.utils import create_path
from lib.utils import cvt_conf
from lib.lammps import get_last_dump

def _gen_lammps_relax (conf_file, 
                       mass_map, 
                       model, 
                       pres, 
                       thermo_freq = 100, 
                       dump_freq = 100) :
    ret = ''
    ret += 'clear\n'
    ret += '# --------------------- VARIABLES-------------------------\n'
    ret += 'variable        THERMO_FREQ     equal %d\n' % thermo_freq
    ret += 'variable        DUMP_FREQ       equal %d\n' % dump_freq
    ret += 'variable        PRES            equal %f\n' % pres
    ret += '# ---------------------- INITIALIZAITION ------------------\n'
    ret += 'units           metal\n'
    ret += 'boundary        p p p\n'
    ret += 'atom_style      atomic\n'
    ret += '# --------------------- ATOM DEFINITION ------------------\n'
    ret += 'box             tilt large\n'
    ret += 'read_data       %s\n' % conf_file
    ret += 'change_box      all triclinic\n'
    for jj in range(len(mass_map)) :
        ret+= "mass            %d %f\n" %(jj+1, mass_map[jj])
    ret += '# --------------------- FORCE FIELDS ---------------------\n'
    ret += 'pair_style      deepmd %s\n' % model
    ret += 'pair_coeff\n'
    ret += '# --------------------- MD SETTINGS ----------------------\n'
    ret += 'neighbor        1.0 bin\n'
    ret += 'thermo          ${THERMO_FREQ}\n'
    ret += 'thermo_style    custom step pe enthalpy press vol pxx pyy pzz pxy pyz pxz\n'
    ret += 'dump		1 all custom 100 dump.relax id type x y z vx vy vz fx fy fz\n'
    ret += '# --------------------- RUN ------------------------------\n'
    ret += 'min_style       cg\n'
    ret += 'fix             1 all box/relax iso ${PRES}\n'
    ret += 'minimize        1.000000e-12 1.000000e-12 500000 500000\n'
    ret += 'fix             1 all box/relax tri ${PRES}\n'
    ret += 'minimize        1.000000e-12 1.000000e-12 500000 500000\n'
    return ret

def make_task(iter_name, jdata, pres) :
    equi_conf = jdata['equi_conf']
    equi_conf = os.path.abspath(equi_conf)
    model = jdata['model']
    model = os.path.abspath(model)
    model_mass_map = jdata['model_mass_map']
    if pres == None :
        pres = jdata['pres']
    elif 'pres' in jdata :
        print('P = %f overrides the pres in json data' % pres)
    jdata['pres'] = pres
    
    create_path(iter_name)
    cwd = os.getcwd()
    os.chdir(iter_name)
    with open('in.json', 'w') as fp:
        json.dump(jdata, fp, indent=4)
    os.symlink(os.path.relpath(equi_conf), 'conf.lmp')
    os.symlink(os.path.relpath(model), 'graph.pb')
    lmp_str \
        = _gen_lammps_relax('conf.lmp',
                            model_mass_map, 
                            'graph.pb',
                            pres)
    with open('in.lammps', 'w') as fp :
        fp.write(lmp_str)
    os.chdir(cwd)

def extract(iter_name, output) :
    if os.path.exists(output) :
        raise RuntimeError('existing file ' + output + ' do nothing')
    dump_file = os.path.join(iter_name, 'dump.relax')
    with open('dump.tmp', 'w') as fp :
        fp.write(get_last_dump(dump_file))
    cvt_conf('dump.tmp', output, ofmt = 'lammps_data')
    os.remove('dump.tmp')

def compute(iter_name) :
    fname = os.path.join(iter_name, 'log.lammps')
    lines = open(fname).read().split('\n')
    for ii in range(len(lines)) :
        if 'Loop time of' in lines[ii] :
            idx = ii-1
            break
    res = [float(jj) for jj in lines[idx].split()]
    return res[1], res[2]

def _main ():
    parser = argparse.ArgumentParser(
        description="Relax conf")
    subparsers = parser.add_subparsers(title='Valid subcommands', dest='command')

    parser_gen = subparsers.add_parser('gen', help='Generate a job')
    parser_gen.add_argument('PARAM', type=str ,
                            help='json parameter file')
    parser_gen.add_argument('-p','--pressure', type=float,
                            help='the pressure of the system')
    parser_gen.add_argument('-o','--output', type=str, default = 'new_job',
                            help='the output folder for the job')

    parser_comp = subparsers.add_parser('extract', help= 'Extract the conf')
    parser_comp.add_argument('JOB', type=str ,
                             help='folder of the job')
    parser_comp.add_argument('-o','--output', type=str, default = 'conf.lmp',
                             help='output conf file name')

    parser_comp = subparsers.add_parser('compute', help= 'Compute the result of a job')
    parser_comp.add_argument('JOB', type=str ,
                             help='folder of the job')
    parser_comp.add_argument('-t','--type', type=str, default = 'helmholtz', 
                             choices=['helmholtz', 'gibbs'], 
                             help='the type of free energy')

    args = parser.parse_args()

    
    if args.command is None :
        parser.print_help()
        exit
    if args.command == 'gen' :
        jdata = json.load(open(args.PARAM, 'r'))        
        make_task(args.output, jdata, args.pressure)
    elif args.command == 'extract' :
        extract(args.JOB, args.output)
    elif args.command == 'compute' :
        ener, enthalpy = compute(args.JOB)
        if args.type == 'helmholtz' :
            print('# Helmholtz free ener (err) [eV] at 0K == energy:')
            print(ener)
        if args.type == 'gibbs' :
            print('# Gibbs free ener (err) [eV] at 0K == ethalpy:')
            print(enthalpy)
        

if __name__ == '__main__' :
    _main()
