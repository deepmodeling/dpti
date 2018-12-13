#!/usr/bin/env python3

import os, sys, json, argparse, glob
import numpy as np
import scipy.constants as pc

import einstein
import lib.lmp as lmp
import lib.water as water
from lib.utils import create_path
from lib.utils import copy_file_list
from lib.utils import block_avg
from lib.utils import integrate
from lib.utils import parse_seq
from lib.lammps import get_thermo

def _ff_angle_on(lamb,
                 model, 
                 bond_k, bond_l,
                 angle_k, angle_t) :
    ret = ''
    ret += 'pair_style      zero 3.0\n'
    ret += 'pair_coeff      * *\n'
    ret += 'bond_style      harmonic\n'
    ret += 'bond_coeff      1 %f %f\n' % (bond_k, bond_l)
    ret += 'variable        ANGLE_K equal ${LAMBDA}*%.16e\n' % angle_k
    ret += 'angle_style     harmonic\n'
    ret += 'angle_coeff     1 ${ANGLE_K} %f\n' % (angle_t)    
    ret += 'compute         e_deep all pe pair\n'
    return ret

def _ff_deep_on(lamb,
                 model, 
                 bond_k, bond_l,
                 angle_k, angle_t) :
    ret = ''
    ret += 'pair_style      deepmd %s \n' % model
    ret += 'pair_coeff      \n'
    ret += 'bond_style      harmonic\n'
    ret += 'bond_coeff      1 %f %f\n' % (bond_k, bond_l)
    ret += 'angle_style     harmonic\n'
    ret += 'angle_coeff     1 %f %f\n' % (angle_k, angle_t)    
    ret += 'fix             l_deep all adapt 1 pair deepmd scale * * v_LAMBDA\n'
    ret += 'compute         e_deep all pe pair\n'
    return ret

def _ff_bond_angle_off(lamb,
                       model, 
                       bond_k, bond_l,
                       angle_k, angle_t) :
    ret = ''
    ret += 'pair_style      deepmd %s \n' % model
    ret += 'pair_coeff      \n'
    ret += 'variable        BOND_K equal %.16e\n' % (bond_k * (1-lamb))
    ret += 'bond_style      harmonic\n'
    ret += 'bond_coeff      1 ${BOND_K} %f\n' % (bond_l)
    ret += 'variable        ANGLE_K equal %.16e\n' % (angle_k * (1-lamb))
    ret += 'angle_style     harmonic\n'
    ret += 'angle_coeff     1 ${ANGLE_K} %f\n' % (angle_t)    
    ret += 'compute         e_deep all pe pair\n'
    return ret


def _gen_lammps_input (step,
                       conf_file, 
                       mass_map,
                       lamb,
                       model,
                       bond_k,
                       bond_l,
                       angle_k,
                       angle_t,
                       nsteps,
                       dt,
                       ens,
                       temp,
                       pres = 1.0, 
                       tau_t = 0.1,
                       tau_p = 0.5,
                       prt_freq = 100, 
                       copies = None) :
    ret = ''
    ret += 'clear\n'
    ret += '# --------------------- VARIABLES-------------------------\n'
    ret += 'variable        NSTEPS          equal %d\n' % nsteps
    ret += 'variable        THERMO_FREQ     equal %d\n' % prt_freq
    ret += 'variable        DUMP_FREQ       equal %d\n' % prt_freq
    ret += 'variable        TEMP            equal %f\n' % temp
    ret += 'variable        PRES            equal %f\n' % pres
    ret += 'variable        TAU_T           equal %f\n' % tau_t
    ret += 'variable        TAU_P           equal %f\n' % tau_p
    ret += 'variable        LAMBDA          equal %.10e\n' % lamb
    ret += '# ---------------------- INITIALIZAITION ------------------\n'
    ret += 'units           metal\n'
    ret += 'boundary        p p p\n'
    ret += 'atom_style      angle\n'
    ret += '# --------------------- ATOM DEFINITION ------------------\n'
    ret += 'box             tilt large\n'
    ret += 'read_data       %s\n' % conf_file
    if copies is not None :
        ret += 'replicate       %d %d %d\n' % (copies[0], copies[1], copies[2])
    ret += 'change_box      all triclinic\n'
    for jj in range(len(mass_map)) :
        ret+= "mass            %d %f\n" %(jj+1, mass_map[jj])
    ret += '# --------------------- FORCE FIELDS ---------------------\n'
    if step == 'angle_on' :
        ret += _ff_angle_on(lamb, model, bond_k, bond_l, angle_k, angle_t)
    elif step == 'deep_on':
        ret += _ff_deep_on(lamb, model, bond_k, bond_l, angle_k, angle_t)
    elif step == 'bond_angle_off':
        ret += _ff_bond_angle_off(lamb, model, bond_k, bond_l, angle_k, angle_t)
    ret += 'special_bonds   lj/coul 1 1 1 angle no\n'
    ret += '# --------------------- MD SETTINGS ----------------------\n'    
    ret += 'neighbor        1.0 bin\n'
    ret += 'timestep        %s\n' % dt
    ret += 'thermo          ${THERMO_FREQ}\n'
    ret += 'thermo_style    custom step ke pe etotal enthalpy temp press vol ebond eangle c_e_deep\n'
    ret += 'thermo_modify   format 9 %.16e\n'
    ret += 'thermo_modify   format 10 %.16e\n'
    ret += 'thermo_modify   format 11 %.16e\n'
    # ret += 'dump            1 all custom ${DUMP_FREQ} dump.hti id type x y z vx vy vz\n'
    if ens == 'nvt' :
        ret += 'fix             1 all nvt temp ${TEMP} ${TEMP} ${TAU_T}\n'
    elif ens == 'npt-iso' or ens == 'npt':
        ret += 'fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}\n'
    elif ens == 'nve' :
        ret += 'fix             1 all nve\n'
    else :
        raise RuntimeError('unknow ensemble %s\n' % ens)        
    ret += '# --------------------- INITIALIZE -----------------------\n'    
    ret += 'velocity        all create ${TEMP} %d\n' % (np.random.randint(0, 2**16))
    ret += '# --------------------- RUN ------------------------------\n'    
    ret += 'run             ${NSTEPS}\n'    
    return ret


def _make_tasks(iter_name, jdata, step) :
    if step == 'angle_on' :
        all_lambda = parse_seq(jdata['lambda_angle_on'])
        protect_eps = jdata['protect_eps']    
        if all_lambda[0] == 0 :
            all_lambda[0] += protect_eps
    elif step == 'deep_on' :
        all_lambda = parse_seq(jdata['lambda_deep_on'])
        protect_eps = jdata['protect_eps']    
        if all_lambda[0] == 0 :
            all_lambda[0] += protect_eps
    elif step == 'bond_angle_off' :
        all_lambda = parse_seq(jdata['lambda_bond_angle_off'])
        protect_eps = jdata['protect_eps']    
        if all_lambda[-1] == 1 :
            all_lambda[-1] -= protect_eps
    equi_conf = jdata['equi_conf']
    equi_conf = os.path.abspath(equi_conf)    
    model = jdata['model']
    model = os.path.abspath(model)
    model_mass_map = jdata['model_mass_map']
    nsteps = jdata['nsteps']
    dt = jdata['dt']
    bond_k = jdata['bond_k']
    bond_l = jdata['bond_l']
    angle_k = jdata['angle_k']
    angle_t = jdata['angle_t']
    stat_freq = jdata['stat_freq']
    ens = jdata['ens']
    temp = jdata['temp']
    pres = jdata['pres']
    tau_t = jdata['tau_t']
    tau_p = jdata['tau_p']

    create_path(iter_name)
    cwd = os.getcwd()
    os.chdir(iter_name)
    os.symlink(os.path.join('..', 'in.json'), 'in.json')
    os.symlink(os.path.relpath(equi_conf), 'orig.lmp')
    lines = water.add_bonds(open('orig.lmp').read().split('\n'))
    open('conf.lmp', 'w').write('\n'.join(lines))
    os.chdir(cwd)
    for idx in range(len(all_lambda)) :
        work_path = os.path.join(iter_name, 'task.%06d' % idx)
        create_path(work_path)
        os.chdir(work_path)
        os.symlink(os.path.join('..', 'conf.lmp'), 'conf.lmp')
        os.symlink(os.path.relpath(model), 'graph.pb')
        lmp_str \
            = _gen_lammps_input(step,
                                'conf.lmp', 
                                model_mass_map, 
                                all_lambda[idx],
                                'graph.pb',
                                bond_k, bond_l, 
                                angle_k, angle_t, 
                                nsteps, 
                                dt, 
                                ens, 
                                temp, 
                                pres, 
                                tau_t = tau_t,
                                tau_p = tau_p,
                                prt_freq = stat_freq)
        with open('in.lammps', 'w') as fp :
            fp.write(lmp_str)
        with open('lambda.out', 'w') as fp :
            fp.write(str(all_lambda[idx]))
        os.chdir(cwd)

def make_tasks(iter_name, jdata) :
    create_path(iter_name)
    cwd = os.getcwd()
    os.chdir(iter_name)    
    with open('in.json', 'w') as fp:
        json.dump(jdata, fp, indent=4)
    os.chdir(cwd)
    subtask_name = os.path.join(iter_name, '00.angle_on')
    _make_tasks(subtask_name, jdata, 'angle_on')
    subtask_name = os.path.join(iter_name, '01.deep_on')
    _make_tasks(subtask_name, jdata, 'deep_on')
    subtask_name = os.path.join(iter_name, '02.bond_angle_off')
    _make_tasks(subtask_name, jdata, 'bond_angle_off')

def _compute_thermo(fname, stat_skip, stat_bsize) :
    data = get_thermo(fname)
    ea, ee = block_avg(data[:, 3], skip = stat_skip, block_size = stat_bsize)
    ha, he = block_avg(data[:, 4], skip = stat_skip, block_size = stat_bsize)
    ta, te = block_avg(data[:, 5], skip = stat_skip, block_size = stat_bsize)
    pa, pe = block_avg(data[:, 6], skip = stat_skip, block_size = stat_bsize)
    va, ve = block_avg(data[:, 7], skip = stat_skip, block_size = stat_bsize)
    thermo_info = {}
    thermo_info['p'] = pa
    thermo_info['p_err'] = pe
    thermo_info['v'] = va
    thermo_info['v_err'] = ve
    thermo_info['e'] = ea
    thermo_info['e_err'] = ee
    thermo_info['h'] = ha
    thermo_info['h_err'] = he
    thermo_info['t'] = ta
    thermo_info['t_err'] = te
    unit_cvt = 1e5 * (1e-10**3) / pc.electron_volt
    thermo_info['pv'] = pa * va * unit_cvt
    thermo_info['pv_err'] = pe * va * unit_cvt
    return thermo_info

def _post_tasks(iter_name, step) :
    jdata = json.load(open(os.path.join(iter_name, 'in.json')))
    stat_skip = jdata['stat_skip']
    stat_bsize = jdata['stat_bsize']
    all_tasks = glob.glob(os.path.join(iter_name, 'task*'))
    all_tasks.sort()
    ntasks = len(all_tasks)
    
    all_lambda = []
    all_bd_a = []
    all_bd_e = []
    all_ag_a = []
    all_ag_e = []
    all_dp_a = []
    all_dp_e = []

    for ii in all_tasks :
        log_name = os.path.join(ii, 'log.lammps')
        data = get_thermo(log_name)
        np.savetxt(os.path.join(ii, 'data'), data, fmt = '%.6e')
        bd_a, bd_e = block_avg(data[:, 8], skip = stat_skip, block_size = stat_bsize)
        ag_a, ag_e = block_avg(data[:, 9], skip = stat_skip, block_size = stat_bsize)
        dp_a, dp_e = block_avg(data[:,10], skip = stat_skip, block_size = stat_bsize)
        lmda_name = os.path.join(ii, 'lambda.out')
        ll = float(open(lmda_name).read())
        all_lambda.append(ll)
        all_bd_a.append(bd_a)
        all_bd_e.append(bd_e)
        all_ag_a.append(ag_a)
        all_ag_e.append(ag_e)
        all_dp_a.append(dp_a)
        all_dp_e.append(dp_e)

    all_lambda = np.array(all_lambda)
    all_bd_a = np.array(all_bd_a)
    all_bd_e = np.array(all_bd_e)
    all_ag_a = np.array(all_ag_a)
    all_ag_e = np.array(all_ag_e)
    all_dp_a = np.array(all_dp_a)
    all_dp_e = np.array(all_dp_e)
    if step == 'angle_on' :        
        de = all_ag_a / all_lambda
        all_err = np.sqrt(np.square(all_ag_e / all_lambda))
    elif step == 'deep_on' :
        de = all_dp_a / all_lambda
        all_err = np.sqrt(np.square(all_dp_e / all_lambda))
    elif step == 'bond_angle_off' :
        de = - (all_bd_a + all_ag_a) / (1 - all_lambda)
        all_err = np.sqrt(np.square(all_bd_e / (1 - all_lambda)) + np.square(all_ag_e / (1 - all_lambda)))

    all_print = []
    all_print.append(np.arange(len(all_lambda)))
    all_print.append(all_lambda)
    all_print.append(de)
    all_print.append(all_err)
    all_print = np.array(all_print)
    np.savetxt(os.path.join(iter_name, 'hti.out'), 
               all_print.T, 
               fmt = '%.8e', 
               header = 'idx lmbda dU dU_err')

    diff_e, err = integrate(all_lambda, de, all_err)

    thermo_info = _compute_thermo(os.path.join(all_tasks[-1], 'log.lammps'), 
                                  stat_skip, stat_bsize)

    return diff_e, err, thermo_info

def _print_thermo_info(info) :
    ptr = '# thermodynamics\n'
    ptr += '# E (err)  [eV]:  %20.8f %20.8f\n' % (info['e'], info['e_err'])
    ptr += '# H (err)  [eV]:  %20.8f %20.8f\n' % (info['h'], info['h_err'])
    ptr += '# T (err)   [K]:  %20.8f %20.8f\n' % (info['t'], info['t_err'])
    ptr += '# P (err) [bar]:  %20.8f %20.8f\n' % (info['p'], info['p_err'])
    ptr += '# V (err) [A^3]:  %20.8f %20.8f\n' % (info['v'], info['v_err'])
    ptr += '# PV(err)  [eV]:  %20.8f %20.8f' % (info['pv'], info['pv_err'])
    print(ptr)

def compute_ideal_mol(iter_name) :
    jdata = json.load(open(os.path.join(iter_name, 'in.json')))
    mass_map = jdata['model_mass_map']
    conf_lines = open(os.path.join(iter_name, 'orig.lmp')).read().split('\n')
    data_sys = lmp.system_data(conf_lines)
    vol = np.linalg.det(data_sys['cell'])
    temp = jdata['temp']
    kk = jdata['bond_k']
    if 'copies' in jdata :
        ncopies = np.prod(jdata['copies'])
    else :
        ncopies = 1
    natom_vec = [ii * ncopies for ii in data_sys['atom_numbs']]

    # kinetic contribution
    fe = 0
    for ii in range(len(natom_vec)) :
        mass = mass_map[ii]
        natoms = natom_vec[ii]
        lambda_k = einstein.compute_lambda(temp, mass)
        fe += 3 * natoms * np.log(lambda_k)
    natoms_o = natom_vec[0]
    natoms_h = 2 * natoms_o
    natoms = natoms_o + natoms_h
    assert(natoms == sum(natom_vec))
    # spring contribution
    lambda_s = einstein.compute_spring(temp, kk * 1.0)
    fe -= natoms_o * np.log((vol * (pc.angstrom**3)))
    fe += 3 * natoms_h * np.log(lambda_s)
    # N!
    fe += natoms_o * np.log(natoms_o) - natoms_o + 0.5 * np.log(2. * np.pi * natoms_o) 
    fe += natoms_h * np.log(np.sqrt(2))
    # to kbT log Z
    fe *= pc.Boltzmann * temp / pc.electron_volt
    return fe

def post_tasks(iter_name) :
    subtask_name = os.path.join(iter_name, '00.angle_on')
    fe = compute_ideal_mol(subtask_name)
    e0, err0, tinfo0 = _post_tasks(subtask_name, 'angle_on')
    # _print_thermo_info(tinfo)
    # print(e, err)
    subtask_name = os.path.join(iter_name, '01.deep_on')
    e1, err1, tinfo1 = _post_tasks(subtask_name, 'deep_on')
    # _print_thermo_info(tinfo)
    # print(e, err)
    subtask_name = os.path.join(iter_name, '02.bond_angle_off')
    e2, err2, tinfo2 = _post_tasks(subtask_name, 'bond_angle_off')
    # _print_thermo_info(tinfo)
    # print(e, err)
    fe = fe + e0 + e1 + e2
    err = np.sqrt(np.square(err0) + np.square(err1) + np.square(err2))
    return fe, err, tinfo2


def _main ():
    parser = argparse.ArgumentParser(
        description="Compute free energy by Hamiltonian TI")
    subparsers = parser.add_subparsers(title='Valid subcommands', dest='command')

    parser_gen = subparsers.add_parser('gen', help='Generate a job')
    parser_gen.add_argument('PARAM', type=str ,
                            help='json parameter file')
    parser_gen.add_argument('-o','--output', type=str, default = 'new_job',
                            help='the output folder for the job')

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
        make_tasks(output, jdata)
    elif args.command == 'compute' :
        fe, fe_err, thermo_info = post_tasks(args.JOB)
        _print_thermo_info(thermo_info)
        if args.type == 'helmholtz' :
            print('# Helmholtz free ener (err) [eV]:')
            print(fe, fe_err)
        if args.type == 'gibbs' :
            pv = thermo_info['pv']
            pv_err = thermo_info['pv_err']
            e1 = fe + pv
            e1_err = np.sqrt(fe_err**2 + pv_err**2)
            print('# Gibbs free ener (err) [eV]:')
            print(e1, e1_err)        
    
if __name__ == '__main__' :
    _main()
