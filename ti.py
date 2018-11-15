#!/usr/bin/env python3

import os, sys, json, argparse, glob
import numpy as np
import scipy.constants as pc

from lib.utils import create_path
from lib.utils import copy_file_list
from lib.utils import block_avg
from lib.utils import integrate
from lib.lammps import get_thermo
from lib.lammps import get_natoms

def make_iter_name (iter_index) :
    return "task_ti." + ('%04d' % iter_index)

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
                       prt_freq = 100) :
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


def make_tasks(iter_index, jdata) :
    equi_conf = jdata['equi_conf']
    equi_conf = os.path.abspath(equi_conf)
    model = jdata['model']
    model = os.path.abspath(model)
    model_mass_map = jdata['model_mass_map']
    nsteps = jdata['nsteps']
    dt = jdata['dt']
    stat_freq = jdata['stat_freq']
    ens = jdata['ens']
    thermos = jdata['thermos']

    iter_name = make_iter_name(iter_index)
    create_path(iter_name)
    cwd = os.getcwd()
    os.chdir(iter_name)
    with open('in.json', 'w') as fp:
        json.dump(jdata, fp, indent=4)
    os.chdir(cwd)
    for idx,ii in enumerate(thermos) :
        work_path = os.path.join(iter_name, 'task.%06d' % idx)
        create_path(work_path)
        os.chdir(work_path)
        os.symlink(os.path.relpath(equi_conf), 'conf.lmp')
        os.symlink(os.path.relpath(model), 'graph.pb')
        lmp_str \
            = _gen_lammps_input('conf.lmp',
                                model_mass_map, 
                                'graph.pb',
                                nsteps, 
                                dt,
                                ens,
                                ii,
                                prt_freq = stat_freq)
        with open('in.lammps', 'w') as fp :
            fp.write(lmp_str)
        with open('thermo.out', 'w') as fp :
            fp.write(str(ii))
        os.chdir(cwd)

def post_tasks(iter_index, jdata, Eo) :
    equi_conf = jdata['equi_conf']
    natoms = get_natoms(equi_conf)
    stat_skip = jdata['stat_skip']
    stat_bsize = jdata['stat_bsize']
    ens = jdata['ens']

    iter_name = make_iter_name(iter_index)
    all_tasks = glob.glob(os.path.join(iter_name, 'task*'))
    all_tasks.sort()
    ntasks = len(all_tasks)
    
    all_t = []
    all_e = []
    all_e_err = []
    integrand = []
    integrand_err = []
    ener_col = 3
    if 'npt' in ens :
        ener_col = 4

    for ii in all_tasks :
        log_name = os.path.join(ii, 'log.lammps')
        data = get_thermo(log_name)
        ea, ee = block_avg(data[:, ener_col], 
                           skip = stat_skip, 
                           block_size = stat_bsize)
        all_e.append(ea)
        all_e_err.append(ee)
        thermo_name = os.path.join(ii, 'thermo.out')
        tt = float(open(thermo_name).read())
        all_t.append(tt)
        integrand.append(ea / (tt * tt))
        integrand_err.append(ee / (tt * tt))

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
               header = 'idx tt Integrand U U_err')

    diff_e, err = integrate(all_t, integrand, integrand_err)
    e1 = (Eo / (all_t[0]) - diff_e) * all_t[-1]
    print(e1, err)

def _main ():
    parser = argparse.ArgumentParser(
        description="Compute free energy by TI")
    parser.add_argument('JOB_IDX', type=int,
                        help='index of job')
    parser.add_argument('PARAM', type=str,
                        help='json parameter file')
    parser.add_argument('TASK', type=str,
                        help='Can be \'gen\': generate tasks. \n\'compute\': compute the free energy. E0 should be provided.')
    parser.add_argument('-e', '--Eo', type=float, default = 0,
                        help='free energy of starting point')
    args = parser.parse_args()

    with open(args.PARAM) as fp: 
        jdata = json.load(fp)

    job_idx = int(args.JOB_IDX)
    if args.TASK == 'gen' :
        make_tasks(job_idx, jdata)
    elif args.TASK == 'compute' :
        e0 = float(args.Eo)
        post_tasks(job_idx, jdata, e0)
    else :
        raise RuntimeError('unknow task: '+args.TASK)
    # get_thermo('log.lammps')
    
if __name__ == '__main__' :
    _main()
        
