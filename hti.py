#/usr/bin/env python3

import os, sys, json, argparse
import numpy as np

from lib.utils import make_iter_name
from lib.utils import create_path
from lib.utils import copy_file_list
from lib.utils import block_avg

def _gen_lammps_input (conf_file, 
                       mass_map,
                       lamb,
                       model,
                       spring_k,
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
    ret += 'variable        LAMBDA          equal %f\n' % lamb
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
    ret += 'fix             l_spring all spring/self %f\n' % (spring_k * (1 - lamb))
    ret += 'fix_modify      l_spring energy yes\n'
    ret += 'fix             l_deep all adapt 1 pair deepmd scale * * v_LAMBDA\n'
    ret += 'compute         e_deep all pe pair\n'
    ret += '# --------------------- MD SETTINGS ----------------------\n'    
    ret += 'neighbor        1.0 bin\n'
    ret += 'velocity        all create ${TEMP} %d\n' % (np.random.randint(0, 2**16))
    ret += 'timestep        %s\n' % dt
    ret += 'thermo          ${THERMO_FREQ}\n'
    ret += 'thermo_style    custom step ke pe etotal temp press vol f_l_spring c_e_deep\n'
    if ens == 'nvt' :
        ret += 'fix             1 all nvt temp ${TEMP} ${TEMP} ${TAU_T}\n'
    elif ens == 'npt-iso' or ens == 'npt':
        ret += 'fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}\n'
    elif ens == 'nve' :
        ret += 'fix             1 all nve\n'
    else :
        raise RuntimeError('unknow ensemble %s\n' % ens)        
    ret += '\n'
    ret += 'run             ${NSTEPS}\n'
    
    return ret


def _get_lammps_thermo(filename) :
    with open(filename, 'r') as fp :
        fc = fp.read().split('\n')
    for sl in range(len(fc)) :
        if 'Step KinEng PotEng TotEng' in fc[sl] :
            break
    for el in range(len(fc)) :
        if 'Loop time of' in fc[el] :
            break
    data = []
    for ii in range(sl+1, el) :
        data.append([float(jj) for jj in fc[ii].split()])
    data = np.array(data)
    return data

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


def post_task(jdata) :
    iter_index = 0
    iter_name = make_iter_name(iter_index)
    all_tasks = glob.glob(os.path.join(iter_name, 'task*'))
    ntasks = len(all_tasks)
    
    all_lambda = []
    all_es = []
    all_es_err = []
    all_es = []
    all_es_err = []
    # for ii in range(ntasks) :
        


def _main ():
    with open('param.json') as fp: 
        jdata = json.load(fp)

    make_tasks(jdata)
    # _get_lammps_thermo('log.lammps')
    
if __name__ == '__main__' :
    _main()
