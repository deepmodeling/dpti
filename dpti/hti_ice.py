#!/usr/bin/env python3

import os, sys, json, argparse, glob, shutil
import numpy as np
import scipy.constants as pc

from dpti import einstein
from dpti import  hti
from dpti.lib import lmp

def _main ():
    parser = argparse.ArgumentParser(
        description="Compute free energy by Hamiltonian TI")
    subparsers = parser.add_subparsers(title='Valid subcommands', dest='command')

    parser_gen = subparsers.add_parser('gen', help='Generate a job')
    parser_gen.add_argument('PARAM', type=str ,
                            help='json parameter file')
    parser_gen.add_argument('-o','--output', type=str, default = 'new_job',
                            help='the output folder for the job')
    parser_gen.add_argument('-s','--switch', type=str, default = 'one-step',
                            choices = ['one-step', 'two-step', 'three-step'],
                            help='one-step: switching on DP and switching off spring simultanenously.\
                            two-step: 1 switching on DP, 2 switching off spring.\
                            three-step: 1 switching on soft LJ, 2 switching on DP, 3 switching off spring and soft LJ.')

    parser_comp = subparsers.add_parser('compute', help= 'Compute the result of a job')
    parser_comp.add_argument('JOB', type=str ,
                             help='folder of the job')
    parser_comp.add_argument('-t','--type', type=str, default = 'helmholtz', 
                             choices=['helmholtz', 'gibbs'], 
                             help='the type of free energy')
    parser_comp.add_argument('-m','--inte-method', type=str, default = 'inte', 
                             choices=['inte', 'mbar'], 
                             help='the method of thermodynamic integration')
    parser_comp.add_argument('-d','--disorder-corr', action = 'store_true',
                             help='apply disorder correction for ice')
    parser_comp.add_argument('-p','--partial-disorder', type=str,
                             choices = ['3', '5'],
                             help='apply partial disorder correction for ice')
    parser_comp.add_argument('-s','--scheme', type=str, default = 'simpson', 
                             help='the numeric integration scheme')
    parser_comp.add_argument('-S','--shift', type=float, default = 0.0, 
                             help='a constant shift in the energy/mole computation, will be removed from FE')
    parser_comp.add_argument('-g', '--pv', type=float, default = None,
                             help='press*vol value override to calculate Gibbs free energy')
    parser_comp.add_argument('-G', '--pv-err', type=float, default = None,
                             help='press*vol error')

    parser_comp = subparsers.add_parser('refine', help= 'Refine the grid of a job')
    parser_comp.add_argument('-i', '--input', type=str, required=True,
                             help='input job')
    parser_comp.add_argument('-o', '--output', type=str, required=True,
                             help='output job')
    parser_comp.add_argument('-e', '--error', type=float, required=True,
                             help='the error required')
    parser_comp.add_argument('-p', '--print', action = 'store_true',
                             help='print the refinement and exit')
    args = parser.parse_args()

    return exec_args(args=args, parser=parser)

def exec_args(args, parser):
    if args.command is None :
        parser.print_help()
        exit
    if args.command == 'gen' :
        output = args.output
        with open(args.PARAM, 'r') as j:
            jdata = json.load(j)
        if 'crystal' in jdata and jdata['crystal'] == 'frenkel' :
            print('# gen task with Frenkel\'s Einstein crystal')
        else :
            print('# gen task with Vega\'s Einstein molecule')
        hti.make_tasks(output, jdata, 'einstein', args.switch)
    elif args.command == 'refine' :
        hti.refine_task(args.input, args.output, args.error, args.print)        
    elif args.command == 'compute' :
        job = args.JOB
        jdata = json.load(open(os.path.join(job, 'in.json'), 'r'))
        fp_conf = open(os.path.join(args.JOB, 'conf.lmp'))
        sys_data = lmp.to_system_data(fp_conf.read().split('\n'))
        natoms = sum(sys_data['atom_numbs'])
        if 'copies' in jdata :
            natoms *= np.prod(jdata['copies'])
        nmols = natoms // 3
        # compute e0
        if 'crystal' not in jdata :
            jdata['crystal'] = 'vega'
        crystal = jdata['crystal']
        if crystal == 'vega' :
            e0 = einstein.free_energy(job) * 3
        else :
            e0 = einstein.frenkel(job) * 3
        # compute Paulin estimate for disordered entropy
        if args.disorder_corr :
            temp = jdata['temp']
            if args.partial_disorder is not None:
                if args.partial_disorder == '5':
                    pauling_corr = -pc.Boltzmann * temp / pc.electron_volt * 0.3817
                    note_pauling = '(ice5)'
                elif args.partial_disorder == '3':
                    pauling_corr = -pc.Boltzmann * temp / pc.electron_volt * 0.3686
                    note_pauling = '(ice3)'
                else:
                    raise RuntimeError(f'unknow partial_disorder {partial_disorder}')
            else:
                pauling_corr = -pc.Boltzmann * temp / pc.electron_volt * np.log(1.5)
                note_pauling = '      '
            e0 += pauling_corr
        else :
            note_pauling = '      '
            pauling_corr = 0
        # compute integration
        de, de_err, thermo_info = hti.post_tasks(job, jdata, natoms = nmols, method = args.inte_method, scheme = args.scheme)
        # printing
        print_format = '%20.12f  %10.3e  %10.3e'
        hti.print_thermo_info(thermo_info)
        if crystal == 'vega' :
            print('# free ener of Einstein Mole: %20.8f' % (e0))
        else :
            print('# free ener of Einstein Crys: %20.8f' % (e0))
        print('# Pauling corr %s:        %20.8f' % (note_pauling, pauling_corr))
        print(('# fe integration              ' + print_format) \
              % (de, de_err[0], de_err[1]))        
        print( '# fe const shift              %20.12f' % args.shift)
        # if args.type == 'helmholtz' :
        print('# Helmholtz free ener per mol (stat_err inte_err) [eV]:')
        print(print_format % (e0 + de - args.shift, de_err[0], de_err[1]))
        if args.type == 'gibbs' :
            if args.pv is not None:
                pv = args.pv
                print(f"# use manual pv=={pv}")
            else:
                pv = thermo_info['pv']
            if args.pv_err is not None:
                pv_err = args.pv_err
                print(f"# use manual pv_err=={pv_err}")
            else:
                pv_err = thermo_info['pv_err']
            e1 = e0 + de + pv - args.shift
            e1_err = np.sqrt(de_err[0]**2 + pv_err**2)
            print('# Gibbs free ener per mol (stat_err inte_err) [eV]:')
            print(print_format % (e1, e1_err, de_err[1]))


if __name__ == '__main__' :
    _main()
