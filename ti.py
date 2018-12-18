#!/usr/bin/env python3

import os, sys, json, argparse, glob
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

def make_iter_name (iter_index) :
    return "task_ti." + ('%04d' % iter_index)

def parse_seq_ginv (seq) :
    tmp_seq = parse_seq(seq)
    t_begin = tmp_seq[0]
    t_end = tmp_seq[-1]
    ngrid = len(tmp_seq) - 1
    hh = (1/t_end - 1/t_begin) / ngrid
    inv_grid = np.arange(1/t_begin, 1/t_end+0.5*hh, hh)
    inv_grid = 1./inv_grid
    print(len(inv_grid), len(tmp_seq), inv_grid)
    return inv_grid

def _gen_lammps_input (conf_file, 
                       mass_map,
                       model,
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
        ret+= "mass            %d %f\n" %(jj+1, mass_map[jj])
    ret += '# --------------------- FORCE FIELDS ---------------------\n'
    ret += 'pair_style      deepmd %s\n' % model
    ret += 'pair_coeff\n'
    ret += '# --------------------- MD SETTINGS ----------------------\n'    
    ret += 'neighbor        1.0 bin\n'
    ret += 'timestep        %s\n' % dt
    ret += 'thermo          ${THERMO_FREQ}\n'
    if ens == 'nvt' :        
        ret += 'thermo_style    custom step ke pe etotal enthalpy temp press vol\n'
    elif 'npt' in ens :
        ret += 'thermo_style    custom step ke pe etotal enthalpy temp press vol\n'
    else :
        raise RuntimeError('unknow ensemble %s\n' % ens)                
    # ret += 'dump            1 all custom ${DUMP_FREQ} dump.hti id type x y z vx vy vz\n'
    if ens == 'nvt' :
        ret += 'fix             1 all nvt temp ${TEMP} ${TEMP} ${TAU_T}\n'
    elif ens == 'npt-iso' or ens == 'npt':
        ret += 'fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}\n'
    elif ens == 'npt-aniso' :
        ret += 'fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} aniso ${PRES} ${PRES} ${TAU_P}\n'
    elif ens == 'npt-tri' :
        ret += 'fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} tri ${PRES} ${PRES} ${TAU_P}\n'
    elif ens == 'nve' :
        ret += 'fix             1 all nve\n'
    else :
        raise RuntimeError('unknow ensemble %s\n' % ens)        
    ret += '# --------------------- INITIALIZE -----------------------\n'    
    ret += 'velocity        all create ${TEMP} %d\n' % (np.random.randint(0, 2**16))
    ret += '# --------------------- RUN ------------------------------\n'    
    ret += 'run             ${NSTEPS}\n'
    
    return ret


def make_tasks(iter_name, jdata) :
    equi_conf = jdata['equi_conf']
    equi_conf = os.path.abspath(equi_conf)
    copies = None
    if 'copies' in jdata :
        copies = jdata['copies']
    model = jdata['model']
    model = os.path.abspath(model)
    model_mass_map = jdata['model_mass_map']
    nsteps = jdata['nsteps']
    dt = jdata['dt']
    stat_freq = jdata['stat_freq']
    # thermos = jdata['thermos']
    ens = jdata['ens']
    path = jdata['path']
    if 'nvt' in ens :
        if path == 't' :
            temps = parse_seq(jdata['temps'])
            tau_t = jdata['tau_t']
            ntasks = len(temps)
        else :
            raise RuntimeError('supported path of nvt ens is \'t\'')
    elif 'npt' in ens :
        if path == 't' :
            temps = parse_seq(jdata['temps'])
            press = jdata['press']
            ntasks = len(temps)
        elif path == 't-ginv' :
            temps = parse_seq_ginv(jdata['temps'])
            press = jdata['press']
            ntasks = len(temps)
        elif path == 'p' :
            temps = jdata['temps']
            press = parse_seq(jdata['press'])
            ntasks = len(press)
        else :
            raise RuntimeError('supported path of npt ens are \'t\' or \'p\'')
        tau_t = jdata['tau_t']
        tau_p = jdata['tau_p']
    else :
        raise RuntimeError('invalid ens')

    create_path(iter_name)
    cwd = os.getcwd()
    os.chdir(iter_name)
    with open('in.json', 'w') as fp:
        json.dump(jdata, fp, indent=4)
    os.chdir(cwd)
    for ii in range(ntasks) :
        work_path = os.path.join(iter_name, 'task.%06d' % ii)
        create_path(work_path)
        os.chdir(work_path)
        os.symlink(os.path.relpath(equi_conf), 'conf.lmp')
        os.symlink(os.path.relpath(model), 'graph.pb')
        if 'nvt' in ens and path == 't' :
            lmp_str \
                = _gen_lammps_input('conf.lmp',
                                    model_mass_map, 
                                    'graph.pb',
                                    nsteps, 
                                    dt,
                                    ens,
                                    temps[ii],
                                    tau_t = tau_t,
                                    prt_freq = stat_freq, 
                                    copies = copies)
            with open('thermo.out', 'w') as fp :
                fp.write('%f' % temps[ii])
        elif 'npt' in ens and (path == 't' or path == 't-ginv'):
            lmp_str \
                = _gen_lammps_input('conf.lmp',
                                    model_mass_map, 
                                    'graph.pb',
                                    nsteps, 
                                    dt,
                                    ens,
                                    temps[ii],
                                    press,
                                    tau_t = tau_t,
                                    tau_p = tau_p,
                                    prt_freq = stat_freq, 
                                    copies = copies)
            with open('thermo.out', 'w') as fp :
                fp.write('%f' % (temps[ii]))
        elif 'npt' in ens and path == 'p' :
            lmp_str \
                = _gen_lammps_input('conf.lmp',
                                    model_mass_map, 
                                    'graph.pb',
                                    nsteps, 
                                    dt,
                                    ens,
                                    temps,
                                    press[ii],
                                    tau_t = tau_t,
                                    tau_p = tau_p,
                                    prt_freq = stat_freq, 
                                    copies = copies)
            with open('thermo.out', 'w') as fp :
                fp.write('%f' % (press[ii]))
        else:
            raise RuntimeError('invalid ens or path setting' )

        with open('in.lammps', 'w') as fp :
            fp.write(lmp_str)
        os.chdir(cwd)

def _compute_thermo (lmplog, stat_skip, stat_bsize) :
    data = get_thermo(lmplog)
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
    thermo_info['t'] = ta
    thermo_info['t_err'] = te
    thermo_info['h'] = ha
    thermo_info['h_err'] = he
    unit_cvt = 1e5 * (1e-10**3) / pc.electron_volt
    thermo_info['pv'] = pa * va * unit_cvt
    thermo_info['pv_err'] = pe * va * unit_cvt
    return thermo_info

def _print_thermo_info(info, more_head = '') :
    ptr = '# thermodynamics %s\n' % more_head
    ptr += '# E (err)  [eV]:  %20.8f %20.8f\n' % (info['e'], info['e_err'])
    ptr += '# H (err)  [eV]:  %20.8f %20.8f\n' % (info['h'], info['h_err'])
    ptr += '# T (err)   [K]:  %20.8f %20.8f\n' % (info['t'], info['t_err'])
    ptr += '# P (err) [bar]:  %20.8f %20.8f\n' % (info['p'], info['p_err'])
    ptr += '# V (err) [A^3]:  %20.8f %20.8f\n' % (info['v'], info['v_err'])
    ptr += '# PV(err)  [eV]:  %20.8f %20.8f' % (info['pv'], info['pv_err'])
    print(ptr)

def post_tasks(iter_name, jdata, Eo) :
    equi_conf = jdata['equi_conf']
    natoms = get_natoms(equi_conf)
    stat_skip = jdata['stat_skip']
    stat_bsize = jdata['stat_bsize']
    ens = jdata['ens']
    path = jdata['path']

    all_tasks = glob.glob(os.path.join(iter_name, 'task*'))
    all_tasks.sort()
    ntasks = len(all_tasks)
    
    all_t = []
    all_e = []
    all_e_err = []
    integrand = []
    integrand_err = []
    if 'nvt' in ens and path == 't' :
        # TotEng
        stat_col = 3
        print('# TI in NVT along T path')
    elif 'npt' in ens and (path == 't' or path == 't-ginv') :
        # Enthalpy
        stat_col = 4
        print('# TI in NPT along T path')
    elif 'npt' in ens and path == 'p' :
        # volume
        stat_col = 7
        print('# TI in NPT along P path')
    else:
        raise RuntimeError('invalid ens or path setting' )

    for ii in all_tasks :
        log_name = os.path.join(ii, 'log.lammps')
        data = get_thermo(log_name)
        np.savetxt(os.path.join(ii, 'data'), data, fmt = '%.6e')
        ea, ee = block_avg(data[:, stat_col], 
                           skip = stat_skip, 
                           block_size = stat_bsize)
        all_e.append(ea)
        all_e_err.append(ee)
        thermo_name = os.path.join(ii, 'thermo.out')
        tt = float(open(thermo_name).read())
        all_t.append(tt)
        if path == 't' or path == 't-ginv':
            integrand.append(ea / (tt * tt))
            integrand_err.append(ee / (tt * tt))
        elif path == 'p' :
            # cvt from barA^3 to eV
            unit_cvt = 1e5 * (1e-10**3) / pc.electron_volt
            integrand.append(ea * unit_cvt)
            integrand_err.append(ee * unit_cvt)
        else:
            raise RuntimeError('invalid path setting' )

    all_print = []
    all_print.append(np.arange(len(all_tasks)))
    all_print.append(all_t)
    all_print.append(integrand)
    all_print.append(all_e)
    all_print.append(all_e_err)
    all_print = np.array(all_print)
    np.savetxt(os.path.join(iter_name, 'ti.out'), 
               all_print.T, 
               fmt = '%.8e', 
               header = 'idx t/p Integrand U/V U/V_err')

    info0 = _compute_thermo(os.path.join(all_tasks[ 0], 'log.lammps'), stat_skip, stat_bsize)
    info1 = _compute_thermo(os.path.join(all_tasks[-1], 'log.lammps'), stat_skip, stat_bsize)
    _print_thermo_info(info0, 'at start point')
    _print_thermo_info(info1, 'at end point')

    all_temps = []
    all_press = []
    all_fe = []
    all_fe_err = []
    all_fe_sys_err = []
    for ii in range(0, len(all_t)) :
        diff_e, err = integrate(all_t[0:ii+1], integrand[0:ii+1], integrand_err[0:ii+1])
        sys_err = integrate_sys_err(all_t[0:ii+1], integrand[0:ii+1])
        if path == 't' or path == 't-ginv':
            e1 = (Eo / (all_t[0]) - diff_e) * all_t[ii]
            err *= all_t[ii]
            sys_err *= all_t[ii]
            all_temps.append(all_t[ii])
            if 'npt' in ens :
                all_press.append(jdata['press'])
        elif path == 'p':
            e1 = Eo + diff_e        
            all_temps.append(jdata['temps'])
            all_press.append(all_t[ii])
        all_fe.append(e1)
        all_fe_err.append(err)
        all_fe_sys_err.append(sys_err)

    if 'nvt' == ens :
        print('#%8s  %15s  %9s  %9s' % ('T(ctrl)', 'F', 'stat_err', 'inte_err'))
        for ii in range(len(all_temps)) :
            print ('%9.2f  %15.8f  %9.2e  %9.2e' 
                   % (all_temps[ii], all_fe[ii], all_fe_err[ii], all_fe_sys_err[ii]))
    elif 'npt' in ens :
        print('#%8s  %15s  %15s  %9s  %9s' % ('T(ctrl)', 'P(ctrl)', 'F', 'stat_err', 'inte_err'))
        for ii in range(len(all_temps)) :
            print ('%9.2f  %15.8e  %15.8f  %9.2e  %9.2e' 
                   % (all_temps[ii], all_press[ii], all_fe[ii], all_fe_err[ii], all_fe_sys_err[ii]))

    # diff_e, err = integrate(all_t, integrand, integrand_err)
    # if path == 't' :
    #     e1 = (Eo / (all_t[0]) - diff_e) * all_t[-1]
    #     err *= all_t[-1]
    # elif path == 'p' :
    #     e1 = Eo + diff_e        
    # print(e1, err)

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
        make_tasks(output, jdata)
    elif args.command == 'compute' :
        job = args.JOB
        jdata = json.load(open(os.path.join(job, 'in.json'), 'r'))
        e0 = float(args.Eo)
        post_tasks(job, jdata, e0)

    
if __name__ == '__main__' :
    _main()
        
