#!/usr/bin/env python3

import os, sys, json, argparse, glob
import numpy as np

from lib.utils import make_iter_name
from lib.utils import create_path
from lib.utils import copy_file_list
from lib.utils import block_avg
from lib.lammps import get_thermo

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
                       norm_style = 'first') :
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
    ret += 'change_box      all triclinic\n'
    for jj in range(len(mass_map)) :
        ret+= "mass            %d %f\n" %(jj+1, mass_map[jj])
    ret += '# --------------------- FORCE FIELDS ---------------------\n'
    ret += 'pair_style      deepmd %s\n' % model
    ret += 'pair_coeff\n'
    ret += 'fix             l_spring all spring/self %.10e\n' % (spring_k * (1 - lamb))
    ret += 'fix_modify      l_spring energy yes\n'
    ret += 'fix             l_deep all adapt 1 pair deepmd scale * * v_LAMBDA\n'
    ret += 'compute         e_deep all pe pair\n'
    ret += '# --------------------- MD SETTINGS ----------------------\n'    
    ret += 'neighbor        1.0 bin\n'
    ret += 'timestep        %s\n' % dt
    ret += 'thermo          ${THERMO_FREQ}\n'
    ret += 'thermo_style    custom step ke pe etotal temp press vol f_l_spring c_e_deep\n'
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


# ret = _gen_lammps_input('conf.lmp',
#                   [27],
#                   1,
#                   'graph.000.pb',
#                   100,
#                   1000,
#                   0.002,
#                   'nvt',
#                   100)
# print(ret)


def make_tasks(jdata) :
    iter_index = 0
    lambda_str = jdata['lambda'].split(':')
    all_lambda = np.arange(float(lambda_str[0]),
                           float(lambda_str[1]) + float(lambda_str[2]), 
                           float(lambda_str[2]))
    protect_eps = jdata['protect_eps']
    if all_lambda[0] == 0:
        all_lambda[0] += protect_eps
    if all_lambda[-1] == 1:
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
    temp = jdata['temp']

    iter_name = make_iter_name(iter_index)
    create_path(iter_name)
    cwd = os.getcwd()
    for idx,ii in enumerate(all_lambda) :
        work_path = os.path.join(iter_name, 'task.%06d' % idx)
        create_path(work_path)
        os.chdir(work_path)
        os.symlink(os.path.relpath(equi_conf), 'conf.lmp')
        os.symlink(os.path.relpath(model), 'graph.pb')
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
                                prt_freq = stat_freq)
        with open('lambda.out', 'w') as fp :
            fp.write(str(ii))
        with open('in.lammps', 'w') as fp :
            fp.write(lmp_str)
        os.chdir(cwd)


def post_tasks(jdata) :
    stat_skip = jdata['stat_skip']
    stat_bsize = jdata['stat_bsize']
    iter_index = 0
    iter_name = make_iter_name(iter_index)
    all_tasks = glob.glob(os.path.join(iter_name, 'task*'))
    all_tasks.sort()
    ntasks = len(all_tasks)
    
    all_lambda = []
    all_es = []
    all_es_err = []
    all_ed = []
    all_ed_err = []

    for ii in all_tasks :
        log_name = os.path.join(ii, 'log.lammps')
        data = get_thermo(log_name)
        sa, se = block_avg(data[:, 7], skip = stat_skip, block_size = stat_bsize)
        da, de = block_avg(data[:, 8], skip = stat_skip, block_size = stat_bsize)
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
    all_print = []
    all_print.append(np.arange(len(all_lambda)))
    all_print.append(all_lambda)
    all_print.append(de)
    all_print.append(all_ed / all_lambda)
    all_print.append(all_es / (1 - all_lambda))
    all_print.append(all_ed)
    all_print.append(all_es)
    all_print = np.array(all_print)
    np.savetxt(os.path.join(iter_name, 'hti.out'), 
               all_print.T, 
               fmt = '%.8e', 
               header = 'idx lmbda dU Ud Us l*Ud (1-l)*Us')

    diff_e = 0
    for ii in range(ntasks-1) :
        diff_e += 0.5 * (all_lambda[ii+1] - all_lambda[ii]) \
                  * (de[ii+1] + de[ii])
    print(diff_e)

def _main ():
    parser = argparse.ArgumentParser(
        description="Compute free energy of Einstein molecule")
    parser.add_argument('PARAM', type=str,
                        help='json parameter file')
    parser.add_argument('TASK', type=str,
                        help='Can be \'gen\': generate tasks. \n\'compute\': compute the free energy difference')
    args = parser.parse_args()

    with open(args.PARAM) as fp: 
        jdata = json.load(fp)

    if args.TASK == 'gen' :
        make_tasks(jdata)
    elif args.TASK == 'compute' :
        post_tasks(jdata)
    else :
        raise RuntimeError('unknow task: '+args.TASK)
    # get_thermo('log.lammps')
    
if __name__ == '__main__' :
    _main()
