#!/usr/bin/env python3

import os, sys, json, argparse, glob, shutil
import numpy as np
import scipy.constants as pc

import einstein
from lib.utils import create_path
from lib.utils import copy_file_list
from lib.utils import block_avg
from lib.utils import integrate
from lib.utils import integrate_sys_err
from lib.utils import parse_seq
from lib.lammps import get_thermo
from lib.lammps import get_natoms

def make_iter_name (iter_index) :
    return "task_hti." + ('%04d' % iter_index)

def _gen_lammps_input (conf_file, 
                       mass_map,
                       lamb,
                       model,
                       spring_k_,
                       nsteps,
                       dt,
                       ens,
                       temp,
                       pres = 1.0, 
                       tau_t = 0.1,
                       tau_p = 0.5,
                       prt_freq = 100, 
                       copies = None,
                       norm_style = 'first', 
                       switch_style = 'both') :
    spring_k = spring_k_[0]
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
    ret += 'atom_style      atomic\n'
    ret += '# --------------------- ATOM DEFINITION ------------------\n'
    ret += 'box             tilt large\n'
    ret += 'read_data       %s\n' % conf_file
    if copies is not None :
        ret += 'replicate       %d %d %d\n' % (copies[0], copies[1], copies[2])
    ret += 'change_box      all triclinic\n'
    for jj in range(len(mass_map)) :
        ret += "mass            %d %f\n" %(jj+1, mass_map[jj])
    ret += '# --------------------- FORCE FIELDS ---------------------\n'
    ret += 'pair_style      deepmd %s\n' % model
    ret += 'pair_coeff\n'
    if switch_style == 'both' :
        ret += 'fix             l_spring all spring/self %.10e\n' % (spring_k * (1 - lamb))
        ret += 'fix_modify      l_spring energy yes\n'
        ret += 'fix             l_deep all adapt 1 pair deepmd scale * * v_LAMBDA\n'
        ret += 'compute         e_deep all pe pair\n'
    elif switch_style == 'deep_on' :
        ret += 'fix             l_spring all spring/self %.10e\n' % (spring_k)
        ret += 'fix_modify      l_spring energy yes\n'
        ret += 'fix             l_deep all adapt 1 pair deepmd scale * * v_LAMBDA\n'
        ret += 'compute         e_deep all pe pair\n'
    elif switch_style == 'spring_off' :
        ret += 'fix             l_spring all spring/self %.10e\n' % (spring_k * (1 - lamb))
        ret += 'fix_modify      l_spring energy yes\n'
        ret += 'compute         e_deep all pe pair\n'
    else :
        raise RuntimeError('unknow switch_style ' + switch_style)        
    ret += '# --------------------- MD SETTINGS ----------------------\n'    
    ret += 'neighbor        1.0 bin\n'
    ret += 'timestep        %s\n' % dt
    ret += 'thermo          ${THERMO_FREQ}\n'
    ret += 'thermo_style    custom step ke pe etotal enthalpy temp press vol f_l_spring c_e_deep\n'
    ret += 'thermo_modify   format 9 %.16e\n'
    ret += 'thermo_modify   format 10 %.16e\n'
    ret += '# dump            1 all custom ${DUMP_FREQ} dump.hti id type x y z vx vy vz\n'
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
    if norm_style == 'com' :
        ret += 'fix             fc all recenter INIT INIT INIT\n'
        ret += 'fix             fm all momentum 1 linear 1 1 1\n'
        ret += 'velocity        all zero linear\n'
    elif norm_style == 'first' :
        ret += 'group           first id 1\n'
        ret += 'fix             fc first recenter INIT INIT INIT\n'
        ret += 'fix             fm first momentum 1 linear 1 1 1\n'
        ret += 'velocity        first zero linear\n'
    else :
        raise RuntimeError('unknow norm_style ' + norm_style)
    ret += '# --------------------- RUN ------------------------------\n'    
    ret += 'run             ${NSTEPS}\n'
    
    return ret

def _gen_lammps_input_ideal (conf_file, 
                             mass_map,
                             lamb,
                             model,
                             nsteps,
                             dt,
                             ens,
                             temp,
                             pres = 1.0, 
                             tau_t = 0.1,
                             tau_p = 0.5,
                             prt_freq = 100, 
                             copies = None,
                             norm_style = 'first') :
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
    ret += 'variable        ZERO            equal 0\n'
    ret += '# ---------------------- INITIALIZAITION ------------------\n'
    ret += 'units           metal\n'
    ret += 'boundary        p p p\n'
    ret += 'atom_style      atomic\n'
    ret += '# --------------------- ATOM DEFINITION ------------------\n'
    ret += 'box             tilt large\n'
    ret += 'read_data       %s\n' % conf_file
    if copies is not None :
        ret += 'replicate       %d %d %d\n' % (copies[0], copies[1], copies[2])
    ret += 'change_box      all triclinic\n'
    for jj in range(len(mass_map)) :
        ret += "mass            %d %f\n" %(jj+1, mass_map[jj])
    ret += '# --------------------- FORCE FIELDS ---------------------\n'
    ret += 'pair_style      deepmd %s\n' % model
    ret += 'pair_coeff\n'
    ret += 'fix             l_deep all adapt 1 pair deepmd scale * * v_LAMBDA\n'
    ret += 'compute         e_deep all pe pair\n'
    ret += '# --------------------- MD SETTINGS ----------------------\n'    
    ret += 'neighbor        1.0 bin\n'
    ret += 'timestep        %s\n' % dt
    ret += 'thermo          ${THERMO_FREQ}\n'
    ret += 'thermo_style    custom step ke pe etotal enthalpy temp press vol v_ZERO c_e_deep\n'
    ret += 'thermo_modify   format 10 %.16e\n'
    ret += '# dump            1 all custom ${DUMP_FREQ} dump.hti id type x y z vx vy vz\n'
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


def make_tasks(iter_name, jdata, ref, switch_style = 'both') :
    all_lambda = parse_seq(jdata['lambda'])
    protect_eps = jdata['protect_eps']
    if all_lambda[0] == 0 and (switch_style == 'both' or switch_style == 'deep_on'):
        all_lambda[0] += protect_eps
    if all_lambda[-1] == 1 and (switch_style == 'both' or switch_style == 'spring_off'):
        all_lambda[-1] -= protect_eps
    equi_conf = jdata['equi_conf']
    equi_conf = os.path.abspath(equi_conf)
    model = jdata['model']
    model = os.path.abspath(model)
    model_mass_map = jdata['model_mass_map']
    nsteps = jdata['nsteps']
    dt = jdata['dt']
    spring_k = jdata['spring_k']
    stat_freq = jdata['stat_freq']
    copies = None
    if 'copies' in jdata :
        copies = jdata['copies']
    temp = jdata['temp']
    jdata['reference'] = ref
    jdata['switch_style'] = switch_style

    create_path(iter_name)
    copied_conf = os.path.join(os.path.abspath(iter_name), 'conf.lmp')
    shutil.copyfile(equi_conf, copied_conf)
    jdata['equi_conf'] = copied_conf
    linked_model = os.path.join(os.path.abspath(iter_name), 'graph.pb')
    os.symlink(model, linked_model)
    jdata['model'] = linked_model

    cwd = os.getcwd()
    os.chdir(iter_name)
    with open('in.json', 'w') as fp:
        json.dump(jdata, fp, indent=4)
    os.chdir(cwd)
    for idx,ii in enumerate(all_lambda) :
        work_path = os.path.join(iter_name, 'task.%06d' % idx)
        create_path(work_path)
        os.chdir(work_path)
        os.symlink(os.path.relpath(copied_conf), 'conf.lmp')
        os.symlink(os.path.relpath(linked_model), 'graph.pb')
        if ref == 'einstein' :
            lmp_str \
                = _gen_lammps_input('conf.lmp',
                                    model_mass_map, 
                                    ii, 
                                    'graph.pb',
                                    spring_k, 
                                    nsteps, 
                                    dt,
                                    'nvt',
                                    temp,
                                    prt_freq = stat_freq, 
                                    copies = copies,
                                    switch_style = switch_style)
        elif ref == 'ideal' :
            lmp_str \
                = _gen_lammps_input_ideal('conf.lmp',
                                          model_mass_map, 
                                          ii, 
                                          'graph.pb',
                                          nsteps, 
                                          dt,
                                          'nvt',
                                          temp,
                                          prt_freq = stat_freq, 
                                          copies = copies)
        else :
            raise RuntimeError('unknow reference system type ' + ref)
        with open('in.lammps', 'w') as fp :
            fp.write(lmp_str)
        with open('lambda.out', 'w') as fp :
            fp.write(str(ii))
        os.chdir(cwd)

def _compute_thermo(fname, natoms, stat_skip, stat_bsize) :
    data = get_thermo(fname)
    ea, ee = block_avg(data[:, 3], skip = stat_skip, block_size = stat_bsize)
    ha, he = block_avg(data[:, 4], skip = stat_skip, block_size = stat_bsize)
    ta, te = block_avg(data[:, 5], skip = stat_skip, block_size = stat_bsize)
    pa, pe = block_avg(data[:, 6], skip = stat_skip, block_size = stat_bsize)
    va, ve = block_avg(data[:, 7], skip = stat_skip, block_size = stat_bsize)
    thermo_info = {}
    thermo_info['p'] = pa
    thermo_info['p_err'] = pe
    thermo_info['v'] = va / natoms
    thermo_info['v_err'] = ve / np.sqrt(natoms)
    thermo_info['e'] = ea / natoms
    thermo_info['e_err'] = ee / np.sqrt(natoms)
    thermo_info['h'] = ha / natoms
    thermo_info['h_err'] = he / np.sqrt(natoms)
    thermo_info['t'] = ta
    thermo_info['t_err'] = te
    unit_cvt = 1e5 * (1e-10**3) / pc.electron_volt
    thermo_info['pv'] = pa * va * unit_cvt / natoms
    thermo_info['pv_err'] = pe * va * unit_cvt  / np.sqrt(natoms)
    return thermo_info

def post_tasks(iter_name, jdata, natoms = None) :
    stat_skip = jdata['stat_skip']
    stat_bsize = jdata['stat_bsize']
    all_tasks = glob.glob(os.path.join(iter_name, 'task*'))
    all_tasks.sort()
    ntasks = len(all_tasks)
    equi_conf = jdata['equi_conf']
    if natoms == None :
        natoms = get_natoms(equi_conf)
        if 'copies' in jdata :
            natoms *= np.prod(jdata['copies'])
    print('# natoms: %d' % natoms)
    
    all_lambda = []
    all_es = []
    all_es_err = []
    all_ed = []
    all_ed_err = []

    for ii in all_tasks :
        log_name = os.path.join(ii, 'log.lammps')
        data = get_thermo(log_name)
        np.savetxt(os.path.join(ii, 'data'), data, fmt = '%.6e')
        sa, se = block_avg(data[:, 8], skip = stat_skip, block_size = stat_bsize)
        da, de = block_avg(data[:, 9], skip = stat_skip, block_size = stat_bsize)
        sa /= natoms
        se /= np.sqrt(natoms)
        da /= natoms
        de /= np.sqrt(natoms)
        lmda_name = os.path.join(ii, 'lambda.out')
        ll = float(open(lmda_name).read())
        all_lambda.append(ll)
        all_es.append(sa)
        all_ed.append(da)
        all_es_err.append(se)
        all_ed_err.append(de)

    all_lambda = np.array(all_lambda)
    all_es = np.array(all_es)
    all_ed = np.array(all_ed)
    all_es_err = np.array(all_es_err)
    all_ed_err = np.array(all_ed_err)
    de = all_ed / all_lambda - all_es / (1 - all_lambda)
    all_err = np.sqrt(np.square(all_ed_err / all_lambda) + np.square(all_es_err / (1 - all_lambda)))

    all_print = []
    # all_print.append(np.arange(len(all_lambda)))
    all_print.append(all_lambda)
    all_print.append(de)
    all_print.append(all_err)
    all_print.append(all_ed / all_lambda)
    all_print.append(all_es / (1 - all_lambda))
    all_print.append(all_ed_err / all_lambda)
    all_print.append(all_es_err / (1 - all_lambda))
    all_print = np.array(all_print)
    np.savetxt(os.path.join(iter_name, 'hti.out'), 
               all_print.T, 
               fmt = '%.8e', 
               header = 'lmbda dU dU_err Ud Us Ud_err Us_err')

    diff_e, err = integrate(all_lambda, de, all_err)
    sys_err = integrate_sys_err(all_lambda, de)

    thermo_info = _compute_thermo(os.path.join(all_tasks[-1], 'log.lammps'), 
                                  natoms,
                                  stat_skip, stat_bsize)

    return diff_e, [err,sys_err], thermo_info


def print_thermo_info(info) :
    ptr = '# thermodynamics (normalized by natoms)\n'
    ptr += '# E (err)  [eV]:  %20.8f %20.8f\n' % (info['e'], info['e_err'])
    ptr += '# H (err)  [eV]:  %20.8f %20.8f\n' % (info['h'], info['h_err'])
    ptr += '# T (err)   [K]:  %20.8f %20.8f\n' % (info['t'], info['t_err'])
    ptr += '# P (err) [bar]:  %20.8f %20.8f\n' % (info['p'], info['p_err'])
    ptr += '# V (err) [A^3]:  %20.8f %20.8f\n' % (info['v'], info['v_err'])
    ptr += '# PV(err)  [eV]:  %20.8f %20.8f' % (info['pv'], info['pv_err'])
    print(ptr)

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
        make_tasks(output, jdata, args.reference, args.switch)
    elif args.command == 'compute' :
        job = args.JOB
        jdata = json.load(open(os.path.join(job, 'in.json'), 'r'))
        de, de_err, thermo_info = post_tasks(job, jdata)
        print_thermo_info(thermo_info)
        if 'reference' not in jdata :
            jdata['reference'] = 'einstein'
        if jdata['reference'] == 'einstein' :
            e0 = einstein.free_energy(jdata)
            print('# free ener of Einstein Mole: %20.8f' % e0)
        else :
            e0 = einstein.ideal_gas_fe(jdata)
            print('# free ener of ideal gas: %20.8f' % e0)
        print_format = '%20.12f  %10.3e  %10.3e'
        if args.type == 'helmholtz' :
            print('# Helmholtz free ener per atom (stat_err inte_err) [eV]:')
            print(print_format % (e0 + de, de_err[0], de_err[1]))
        if args.type == 'gibbs' :
            pv = thermo_info['pv']
            pv_err = thermo_info['pv_err']
            e1 = e0 + de + pv
            e1_err = np.sqrt(de_err[0]**2 + pv_err**2)
            print('# Gibbs free ener per atom (stat_err inte_err) [eV]:')
            print(print_format % (e1, e1_err, de_err[1]))
    
if __name__ == '__main__' :
    _main()
