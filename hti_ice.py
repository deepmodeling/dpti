#!/usr/bin/env python3

import os, sys, json, argparse, glob, shutil
import numpy as np
import scipy.constants as pc

import einstein
import hti
import lib.lmp as lmp

def _main ():
    parser = argparse.ArgumentParser(
        description="Compute free energy by Hamiltonian TI")
    subparsers = parser.add_subparsers(title='Valid subcommands', dest='command')

    parser_gen = subparsers.add_parser('gen', help='Generate a job')
    parser_gen.add_argument('PARAM', type=str ,
                            help='json parameter file')
    parser_gen.add_argument('-o','--output', type=str, default = 'new_job',
                            help='the output folder for the job')
    parser_gen.add_argument('-r','--reference', type=str, default = 'einstein', 
                            choices=['einstein', 'ideal'], 
                            help='the reference state, einstein crystal or ideal gas')
    parser_gen.add_argument('-s','--switch', type=str, default = 'both', 
                            choices=['both', 'deep_on', 'spring_off'], 
                            help='the reference state, einstein crystal or ideal gas')

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
        output = args.output
        jdata = json.load(open(args.PARAM, 'r'))
        hti.make_tasks(output, jdata, args.reference, args.switch)
    elif args.command == 'compute' :
        job = args.JOB
        jdata = json.load(open(os.path.join(job, 'in.json'), 'r'))
        de, de_err, thermo_info = hti.post_tasks(job, jdata)
        hti.print_thermo_info(thermo_info)
        fp_conf = open(os.path.join(args.JOB, 'conf.lmp'))
        sys_data = lmp.to_system_data(fp_conf.read().split('\n'))
        natoms = sum(sys_data['atom_numbs'])
        if 'ncopies' in jdata :
            natoms *= np.prod(jdata['ncopies'])
        nmols = natoms // 3
        print ('# numb atoms: %d' % natoms)
        print ('# numb  mols: %d' % nmols)
        if 'reference' not in jdata :
            jdata['reference'] = 'einstein'
        if jdata['reference'] == 'einstein' :
            jdata1 = jdata
            jdata1['equi_conf'] = os.path.join(args.JOB, 'conf.lmp')
            e0 = einstein.free_energy(jdata1)
            print('# free ener of Einstein Mole: %20.8f' % e0)
        else :
            e0 = einstein.ideal_gas_fe(jdata)
            print('# free ener of ideal gas: %20.8f' % e0)
        if args.type == 'helmholtz' :
            print('# Helmholtz free ener (err) [eV]:')
            print('%20.8f  %10.3e' % (e0 + de, de_err))
            print('# Helmholtz free ener per mol (err) [eV]:')
            print('%20.8f  %10.3e' % ((e0 + de) / nmols, de_err / np.sqrt(nmols)))
        if args.type == 'gibbs' :
            pv = thermo_info['pv']
            pv_err = thermo_info['pv_err']
            e1 = e0 + de + pv
            e1_err = np.sqrt(de_err**2 + pv_err**2)
            print('# Gibbs free ener (err) [eV]:')
            print('%20.8f  %10.3e' % (e1, e1_err))
            print('# Gibbs free ener per mol (err) [eV]:')
            print('%20.8f  %10.3e' % (e1 / nmols, e1_err / np.sqrt(nmols)))


if __name__ == '__main__' :
    _main()
