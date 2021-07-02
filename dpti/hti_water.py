#!/usr/bin/env python3

import os, sys, json, argparse, glob, shutil
import numpy as np
import scipy.constants as pc

from dpti import einstein
from dpti.lib import lmp
from  dpti.lib import water
import pymbar
from dpti.lib.utils import create_path
from dpti.lib.utils import copy_file_list
from dpti.lib.utils import block_avg
from dpti.lib.utils import integrate_range
# from lib.utils import integrate_sys_err
from dpti.lib.utils import compute_nrefine
from dpti.lib.utils import parse_seq
from dpti.lib.utils import get_task_file_abspath
from dpti.lib.utils import get_first_matched_key_from_dict
from dpti.lib.lammps import get_thermo

def _ff_angle_on(lamb,
                 model, 
                 bparam,
                 sparam) :
    bond_k = bparam['bond_k']
    bond_l = bparam['bond_l']
    angle_k = bparam['angle_k']
    angle_t = bparam['angle_t']
    nn = sparam['n']
    alpha_lj = sparam['alpha_lj']
    rcut = sparam['rcut']
    epsilon = sparam['epsilon']
    sigma_oo = sparam['sigma_oo']
    sigma_oh = sparam['sigma_oh']
    sigma_hh = sparam['sigma_hh']
    activation = sparam['activation']
    ret = ''
    ret += 'variable        EPSILON equal %f\n' % epsilon
    ret += 'pair_style      lj/cut/soft %f %f %f  \n' % (nn, alpha_lj, rcut)
    ret += 'pair_coeff      1 1 ${EPSILON} %f %f\n' % (sigma_oo, activation)
    ret += 'pair_coeff      1 2 ${EPSILON} %f %f\n' % (sigma_oh, activation)
    ret += 'pair_coeff      2 2 ${EPSILON} %f %f\n' % (sigma_hh, activation)
    ret += 'bond_style      harmonic\n'
    ret += 'bond_coeff      1 %f %f\n' % (bond_k, bond_l)
    ret += 'variable        ANGLE_K equal ${LAMBDA}*%.16e\n' % angle_k
    ret += 'angle_style     harmonic\n'
    ret += 'angle_coeff     1 ${ANGLE_K} %f\n' % (angle_t)    
    ret += 'fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_LAMBDA scale yes\n'
    ret += 'compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_EPSILON\n'    
    return ret

def _ff_deep_on(lamb,
                 model, 
                 bparam,
                 sparam) :
    bond_k = bparam['bond_k']
    bond_l = bparam['bond_l']
    angle_k = bparam['angle_k']
    angle_t = bparam['angle_t']
    nn = sparam['n']
    alpha_lj = sparam['alpha_lj']
    rcut = sparam['rcut']
    epsilon = sparam['epsilon']
    sigma_oo = sparam['sigma_oo']
    sigma_oh = sparam['sigma_oh']
    sigma_hh = sparam['sigma_hh']
    activation = sparam['activation']
    ret = ''
    ret += 'variable        EPSILON equal %f\n' % epsilon
    ret += 'variable        ONE equal 1\n'
    ret += 'pair_style      hybrid/overlay deepmd %s lj/cut/soft %f %f %f  \n' % (model, nn, alpha_lj, rcut)
    ret += 'pair_coeff      * * deepmd\n'
    ret += 'pair_coeff      1 1 lj/cut/soft ${EPSILON} %f %f\n' % (sigma_oo, activation)
    ret += 'pair_coeff      1 2 lj/cut/soft ${EPSILON} %f %f\n' % (sigma_oh, activation)
    ret += 'pair_coeff      2 2 lj/cut/soft ${EPSILON} %f %f\n' % (sigma_hh, activation)
    ret += 'bond_style      harmonic\n'
    ret += 'bond_coeff      1 %f %f\n' % (bond_k, bond_l)
    ret += 'angle_style     harmonic\n'
    ret += 'angle_coeff     1 %f %f\n' % (angle_k, angle_t)    
    ret += 'fix             tot_pot all adapt/fep 0 pair deepmd scale * * v_LAMBDA\n'
    ret += 'compute         e_diff all fep ${TEMP} pair deepmd scale * * v_ONE\n'
    return ret

def _ff_bond_angle_off(lamb,
                       model, 
                       bparam,
                       sparam) :
    bond_k = bparam['bond_k']
    bond_l = bparam['bond_l']
    angle_k = bparam['angle_k']
    angle_t = bparam['angle_t']
    nn = sparam['n']
    alpha_lj = sparam['alpha_lj']
    rcut = sparam['rcut']
    epsilon = sparam['epsilon']
    sigma_oo = sparam['sigma_oo']
    sigma_oh = sparam['sigma_oh']
    sigma_hh = sparam['sigma_hh']
    activation = sparam['activation']
    ret = ''
    ret += 'variable        INV_LAMBDA equal 1-${LAMBDA}\n'
    ret += 'variable        EPSILON equal %f\n' % epsilon
    ret += 'variable        INV_EPSILON equal -${EPSILON}\n'
    ret += 'pair_style      hybrid/overlay deepmd %s lj/cut/soft %f %f %f  \n' % (model, nn, alpha_lj, rcut)
    ret += 'pair_coeff      * * deepmd\n'
    ret += 'pair_coeff      1 1 lj/cut/soft ${EPSILON} %f %f\n' % (sigma_oo, activation)
    ret += 'pair_coeff      1 2 lj/cut/soft ${EPSILON} %f %f\n' % (sigma_oh, activation)
    ret += 'pair_coeff      2 2 lj/cut/soft ${EPSILON} %f %f\n' % (sigma_hh, activation)
    ret += 'variable        BOND_K equal %.16e\n' % (bond_k * (1-lamb))
    ret += 'bond_style      harmonic\n'
    ret += 'bond_coeff      1 ${BOND_K} %f\n' % (bond_l)
    ret += 'variable        ANGLE_K equal %.16e\n' % (angle_k * (1-lamb))
    ret += 'angle_style     harmonic\n'
    ret += 'angle_coeff     1 ${ANGLE_K} %f\n' % (angle_t)    
    ret += 'fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_INV_LAMBDA scale yes\n'
    ret += 'compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_INV_EPSILON\n'    
    return ret


def _gen_lammps_input (step,
                       conf_file, 
                       mass_map,
                       lamb,
                       model,
                       bparam,
                       sparam,
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
        ret += _ff_angle_on(lamb, model, bparam, sparam)
    elif step == 'deep_on':
        ret += _ff_deep_on(lamb, model, bparam, sparam)
    elif step == 'bond_angle_off':
        ret += _ff_bond_angle_off(lamb, model, bparam, sparam)
    ret += 'special_bonds   lj/coul 1 1 1 angle no\n'
    ret += '# --------------------- MD SETTINGS ----------------------\n'    
    ret += 'neighbor        1.0 bin\n'
    ret += 'timestep        %s\n' % dt
    ret += 'thermo          ${THERMO_FREQ}\n'
    ret += 'thermo_style    custom step ke pe etotal enthalpy temp press vol ebond eangle c_e_diff[1]\n'
    ret += 'thermo_modify   format 9 %.16e\n'
    ret += 'thermo_modify   format 10 %.16e\n'
    ret += 'thermo_modify   format 11 %.16e\n'
    ret += '# dump            1 all custom ${DUMP_FREQ} dump.hti id type x y z\n'
    if ens == 'nvt' :
        ret += 'fix             1 all nvt temp ${TEMP} ${TEMP} ${TAU_T}\n'
    elif ens == 'npt-iso' or ens == 'npt':
        ret += 'fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}\n'
    elif ens == 'nve' :
        ret += 'fix             1 all nve\n'
    else :
        raise RuntimeError('unknow ensemble %s\n' % ens)        
    ret += 'fix             mzero all momentum 10 linear 1 1 1\n'
    ret += '# --------------------- INITIALIZE -----------------------\n'    
    ret += 'velocity        all create ${TEMP} %d\n' % (np.random.randint(1, 2**16))
    ret += 'velocity        all zero linear\n'
    ret += '# --------------------- RUN ------------------------------\n'    
    ret += 'run             ${NSTEPS}\n'    
    ret += 'write_data      out.lmp\n'

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
    # mass_map = jdata['mass_map']
    mass_map = get_first_matched_key_from_dict(jdata, ['mass_map', 'model_mass_map'])
    nsteps = jdata['nsteps']
    # dt = jdata['dt']
    # timestep = jdata['timestep']
    timestep = get_first_matched_key_from_dict(jdata, ['timestep', 'dt'])
    bparam = jdata['bond_param']
    sparam = jdata['soft_param']
    # stat_freq = jdata['stat_freq']
    # thermo_freq = jdata['thermo_freq']
    thermo_freq = get_first_matched_key_from_dict(jdata, ['thermo_freq', 'stat_freq'])
    ens = jdata['ens']
    temp = jdata['temp']
    pres = jdata['pres']
    tau_t = jdata['tau_t']
    tau_p = jdata['tau_p']
    copies = None
    if 'copies' in jdata :
        copies = jdata['copies']

    create_path(iter_name)
    cwd = os.getcwd()
    os.chdir(iter_name)
    os.symlink(os.path.join('..', 'in.json'), 'in.json')
    os.symlink(os.path.join('..', 'conf.lmp'), 'orig.lmp')
    os.symlink(os.path.join('..', 'graph.pb'), 'graph.pb')
    with open('orig.lmp', 'r') as f:
        lines = water.add_bonds(f.read().split('\n'))
    with open('conf.lmp', 'w') as c:
        c.write('\n'.join(lines))
    os.chdir(cwd)
    for idx in range(len(all_lambda)) :
        work_path = os.path.join(iter_name, 'task.%06d' % idx)
        create_path(work_path)
        os.chdir(work_path)
        os.symlink(os.path.join('..', 'conf.lmp'), 'conf.lmp')
        os.symlink(os.path.join('..', 'graph.pb'), 'graph.pb')
        lmp_str \
            = _gen_lammps_input(step,
                                'conf.lmp', 
                                mass_map, 
                                all_lambda[idx],
                                'graph.pb',
                                bparam,
                                sparam,
                                nsteps, 
                                timestep, 
                                ens, 
                                temp, 
                                pres, 
                                tau_t = tau_t,
                                tau_p = tau_p,
                                prt_freq = thermo_freq, 
                                copies = copies)
        with open('in.lammps', 'w') as fp :
            fp.write(lmp_str)
        with open('lambda.out', 'w') as fp :
            fp.write(str(all_lambda[idx]))
        os.chdir(cwd)

def _refine_tasks(from_task, to_task, err, step) :
    from_task = os.path.abspath(from_task)
    to_task = os.path.abspath(to_task)

    from_ti = os.path.join(from_task, 'hti.out')
    if not os.path.isfile(from_ti) :
        raise RuntimeError("cannot find file %s, task should be computed befor refined" % from_ti)
    tmp_array = np.loadtxt(from_ti)
    all_t = tmp_array[:,0]
    integrand = tmp_array[:,1]
    ntask = all_t.size
    
    interval_nrefine = compute_nrefine(all_t, integrand, err)

    refined_t = []
    back_map = []
    for ii in range(0, ntask-1) :
        refined_t.append(all_t[ii])
        back_map.append(ii)
        hh = (all_t[ii+1] - all_t[ii]) / interval_nrefine[ii]
        for jj in range(1, interval_nrefine[ii]) :
            refined_t.append(all_t[ii] + jj * hh)
            back_map.append(-1)
    refined_t.append(all_t[-1])
    back_map.append(ntask-1)
    
    from_json = os.path.join(from_task, 'in.json')
    to_json = os.path.join(to_task, 'in.json')
    from_jdata = json.load(open(from_json))
    to_jdata = from_jdata

    to_jdata['orig_task'] = from_task
    to_jdata['back_map'] = back_map
    to_jdata['refine_error'] = err
    if step == 'angle_on' :
        to_jdata['lambda_angle_on'] = refined_t
        to_jdata['lambda_angle_on_back_map'] = back_map
    elif step == 'deep_on' :
        to_jdata['lambda_deep_on'] = refined_t
        to_jdata['lambda_deep_on_back_map'] = back_map
    elif step == 'bond_angle_off' :
        to_jdata['lambda_bond_angle_off'] = refined_t
        to_jdata['lambda_bond_angle_off_back_map'] = back_map
    else :
        raise RuntimeError('unknow step')

    _make_tasks(to_task, to_jdata, step)

    from_task_list = glob.glob(os.path.join(from_task, 'task.[0-9]*'))
    from_task_list.sort()
    to_task_list = glob.glob(os.path.join(to_task, 'task.[0-9]*'))
    to_task_list.sort()
    assert(len(from_task_list) == ntask)
    assert(len(to_task_list) == len(refined_t))

    for ii in range(len(to_task_list)) :
        if back_map[ii] < 0 : 
            continue
        for jj in ['data', 'log.lammps'] :
            shutil.copyfile(
                os.path.join(from_task_list[back_map[ii]], jj), 
                os.path.join(to_task_list[ii], jj), 
            )
        with open(os.path.join(to_task_list[ii], 'from.dir'), 'w') as fp:
            fp.write(from_task_list[back_map[ii]])    


def make_tasks(iter_name, jdata) :
    equi_conf = os.path.abspath(jdata['equi_conf'])
    model = os.path.abspath(jdata['model'])

    create_path(iter_name)
    copied_conf = os.path.join(os.path.abspath(iter_name), 'conf.lmp')
    shutil.copyfile(equi_conf, copied_conf)
    jdata['equi_conf'] = 'conf.lmp'
    linked_model = os.path.join(os.path.abspath(iter_name), 'graph.pb')
    shutil.copyfile(model, linked_model)
    jdata['model'] = 'graph.pb'

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


def refine_tasks(from_task, to_task, err) :
    jdata = json.load(open(os.path.join(from_task, 'in.json')))    
    equi_conf = get_task_file_abspath(from_task, jdata['equi_conf'])
    model = get_task_file_abspath(from_task, jdata['model'])

    create_path(to_task)
    copied_conf = os.path.join(os.path.abspath(to_task), 'conf.lmp')
    shutil.copyfile(equi_conf, copied_conf)
    jdata['equi_conf'] = 'conf.lmp'
    linked_model = os.path.join(os.path.abspath(to_task), 'graph.pb')
    shutil.copyfile(model, linked_model)
    jdata['model'] = 'graph.pb'
    jdata['orig_task'] = from_task
    jdata['refine_error'] = err

    cwd = os.getcwd()
    os.chdir(to_task)
    with open('in.json', 'w') as fp:
        json.dump(jdata, fp, indent=4)
    os.chdir(cwd)

    from_name = os.path.join(from_task, '00.angle_on')
    to_name = os.path.join(to_task, '00.angle_on')
    _refine_tasks(from_name, to_name, err, 'angle_on')
    from_name = os.path.join(from_task, '01.deep_on')
    to_name = os.path.join(to_task, '01.deep_on')
    _refine_tasks(from_name, to_name, err, 'deep_on')
    from_name = os.path.join(from_task, '02.bond_angle_off')
    to_name = os.path.join(to_task, '02.bond_angle_off')
    _refine_tasks(from_name, to_name, err, 'bond_angle_off')


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

def _post_tasks(iter_name, step, natoms, scheme = 's') :
    jdata = json.load(open(os.path.join(iter_name, 'in.json')))
    stat_skip = jdata['stat_skip']
    stat_bsize = jdata['stat_bsize']
    all_tasks = glob.glob(os.path.join(iter_name, 'task.[0-9]*'))
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
        bd_a /= natoms
        ag_a /= natoms
        dp_a /= natoms
        bd_e /= np.sqrt(natoms)
        ag_e /= np.sqrt(natoms)
        dp_e /= np.sqrt(natoms)
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
        de = all_ag_a / all_lambda + all_dp_a
        all_err = np.sqrt(np.square(all_ag_e / all_lambda) +
                          np.square(all_dp_e))
    elif step == 'deep_on' :
        de = all_dp_a
        all_err = all_dp_e
    elif step == 'bond_angle_off' :
        de = - (all_bd_a + all_ag_a) / (1 - all_lambda) + all_dp_a
        all_err = np.sqrt(np.square(all_bd_e / (1 - all_lambda)) + 
                          np.square(all_ag_e / (1 - all_lambda)) + 
                          np.square(all_dp_e))

    all_print = []
    # all_print.append(np.arange(len(all_lambda)))
    all_print.append(all_lambda)
    all_print.append(de)
    all_print.append(all_err)
    all_print = np.array(all_print)
    np.savetxt(os.path.join(iter_name, 'hti.out'), 
               all_print.T, 
               fmt = '%.8e', 
               header = 'lmbda dU dU_err')

    new_lambda, i, i_e, s_e = integrate_range(all_lambda, de, all_err, scheme = scheme)
    if new_lambda[-1] != all_lambda[-1] :
        if new_lambda[-1] == all_lambda[-2]:
            _, i1, i_e1, s_e1 = integrate_range(all_lambda[-2:], de[-2:], all_err[-2:], scheme='t')
            diff_e = i[-1] + i1[-1]
            err = np.linalg.norm([s_e[-1], s_e1[-1]])
            sys_err = i_e[-1] + i_e1[-1]
        else :
            raise RuntimeError("lambda does not match!")
    else:
        diff_e = i[-1]
        err = s_e[-1]
        sys_err = i_e[-1]

    # diff_e, err = integrate(all_lambda, de, all_err)
    # sys_err = integrate_sys_err(all_lambda, de)

    thermo_info = _compute_thermo(os.path.join(all_tasks[-1], 'log.lammps'), 
                                  natoms,
                                  stat_skip, stat_bsize)

    return diff_e, [err, sys_err], thermo_info


def _post_tasks_mbar(iter_name, step, natoms) :
    jdata = json.load(open(os.path.join(iter_name, 'in.json')))
    stat_skip = jdata['stat_skip']
    stat_bsize = jdata['stat_bsize']
    temp = jdata['temp']
    all_tasks = glob.glob(os.path.join(iter_name, 'task.[0-9]*'))
    all_tasks.sort()
    ntasks = len(all_tasks)

    all_lambda = []
    for ii in all_tasks :
        lmda_name = os.path.join(ii, 'lambda.out')
        ll = float(open(lmda_name).read())
        all_lambda.append(ll)
    all_lambda = np.array(all_lambda)
    nlambda = all_lambda.size

    ukn = np.array([])
    nk = []
    kt_in_ev = pc.Boltzmann * temp / pc.electron_volt
    for idx,ii in enumerate(all_tasks) :
        log_name = os.path.join(ii, 'log.lammps')
        data = get_thermo(log_name)
        np.savetxt(os.path.join(ii, 'data'), data, fmt = '%.6e')
        bd_e = data[stat_skip:, 8]/kt_in_ev
        ag_e = data[stat_skip:, 9]/kt_in_ev
        dp_e = data[stat_skip:,10]/kt_in_ev
        if step == 'angle_on' :        
            de = ag_e / all_lambda[idx] + dp_e
        elif step == 'deep_on' :
            de = dp_e
        elif step == 'bond_angle_off' :
            de = -(bd_e + ag_e) / (1 - all_lambda[idx]) + dp_e
        else :
            raise RuntimeError("unknow step")
        nk.append(de.size)
        block_u = []
        for ll in all_lambda :
            if step == 'angle_on' or 'deep_on':
                block_u.append(de * ll)
            else :
                block_u.append(-de * (1-ll))
        block_u = np.reshape(block_u, [nlambda, -1])
        if ukn.size == 0 :
            ukn = block_u 
        else :
            ukn = np.concatenate((ukn, block_u), axis = 1)

    nk = np.array(nk)
    mbar = pymbar.MBAR(ukn, nk)
    Deltaf_ij, dDeltaf_ij, Theta_ij = mbar.getFreeEnergyDifferences()
    Deltaf_ij = Deltaf_ij / natoms
    dDeltaf_ij = dDeltaf_ij / np.sqrt(natoms)

    diff_e = Deltaf_ij[0,-1] * kt_in_ev
    err = dDeltaf_ij[0,-1] * kt_in_ev
    sys_err = 0

    thermo_info = _compute_thermo(os.path.join(all_tasks[-1], 'log.lammps'), 
                                  natoms,
                                  stat_skip, stat_bsize)

    return diff_e, [err, sys_err], thermo_info


def _print_thermo_info(info) :
    ptr = '# thermodynamics (normalized by nmols)\n'
    ptr += '# E (err)  [eV]:  %20.8f %20.8f\n' % (info['e'], info['e_err'])
    ptr += '# H (err)  [eV]:  %20.8f %20.8f\n' % (info['h'], info['h_err'])
    ptr += '# T (err)   [K]:  %20.8f %20.8f\n' % (info['t'], info['t_err'])
    ptr += '# P (err) [bar]:  %20.8f %20.8f\n' % (info['p'], info['p_err'])
    ptr += '# V (err) [A^3]:  %20.8f %20.8f\n' % (info['v'], info['v_err'])
    ptr += '# PV(err)  [eV]:  %20.8f %20.8f' % (info['pv'], info['pv_err'])
    print(ptr)

def spring_inte(temp, kk, r0) :
    kto2k = pc.Boltzmann * temp / (2. * kk * pc.electron_volt / (pc.angstrom * pc.angstrom))
    # print((r0 * pc.angstrom), np.sqrt(kto2k))
    return 4 * np.pi * np.sqrt(2 * np.pi * kto2k) * ((r0 * pc.angstrom) ** 2 + kto2k)

def compute_ideal_mol(iter_name) :
    jdata = json.load(open(os.path.join(iter_name, 'in.json')))
    ens = jdata['ens']
    mass_map = jdata['mass_map']
    conf_lines = open(os.path.join(iter_name, 'orig.lmp')).read().split('\n')
    data_sys = lmp.system_data(conf_lines)
    vol = np.linalg.det(data_sys['cell'])
    temp = jdata['temp']    
    kk = jdata['bond_param']['bond_k']
    ll = jdata['bond_param']['bond_l']
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
    lambda_s = einstein.compute_spring(temp, kk * 2.0)
    lambda_s1 = spring_inte(temp, kk, ll)
    fe -= natoms_o * np.log((vol * (pc.angstrom**3)))
    # print((1/lambda_s))
    # fe += 3 * natoms_h * np.log(lambda_s)
    fe -= natoms_h * np.log(lambda_s1)
    # N!
    fe += natoms_o * np.log(natoms_o) - natoms_o + 0.5 * np.log(2. * np.pi * natoms_o) 
    fe += natoms_h * np.log(np.sqrt(2))
    # plus PV
    if 'npt' in ens :
        fe += natoms_o + 5./6. * natoms_h
    # to kbT log Z
    fe *= pc.Boltzmann * temp / pc.electron_volt
    return fe / natoms_o

def post_tasks(iter_name, natoms, method = 'inte', scheme = 's') :
    subtask_name = os.path.join(iter_name, '00.angle_on')
    fe = compute_ideal_mol(subtask_name)
    print('# fe of ideal mol: %20.12f' % fe)
#    print('# fe of ideal gas: %20.12f' % (einstein.ideal_gas_fe(subtask_name) * 3))
    if method == 'inte' :
        e0, err0, tinfo0 = _post_tasks(subtask_name, 'angle_on', natoms, scheme=scheme)
    elif method == 'mbar' :
        e0, err0, tinfo0 = _post_tasks_mbar(subtask_name, 'angle_on', natoms)
    print('# fe of angle_on : %20.12f  %10.3e %10.3e' % (e0, err0[0], err0[1]))
    # _print_thermo_info(tinfo)
    # print(e, err)
    subtask_name = os.path.join(iter_name, '01.deep_on')
    if method == 'inte' :
        e1, err1, tinfo1 = _post_tasks(subtask_name, 'deep_on', natoms, scheme=scheme)
    elif method == 'mbar' :
        e1, err1, tinfo1 = _post_tasks_mbar(subtask_name, 'deep_on', natoms)
    print('# fe of deep_on  : %20.12f  %10.3e %10.3e' % (e1, err1[0], err1[1]))
    # _print_thermo_info(tinfo)
    # print(e, err)
    subtask_name = os.path.join(iter_name, '02.bond_angle_off')
    if method == 'inte' :
        e2, err2, tinfo2 = _post_tasks(subtask_name, 'bond_angle_off', natoms, scheme=scheme)
    elif method == 'mbar' :
        e2, err2, tinfo2 = _post_tasks_mbar(subtask_name, 'bond_angle_off', natoms)
    print('# fe of bond_off : %20.12f  %10.3e %10.3e' % (e2, err2[0], err2[1]))
    # _print_thermo_info(tinfo)
    # print(e, err)
    fe = fe + e0 + e1 + e2
    err = np.sqrt(np.square(err0[0]) + np.square(err1[0]) + np.square(err2[0]))
    sys_err = ((err0[1]) + (err1[1]) + (err2[1]))
    return fe, [err,sys_err], tinfo2


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
    parser_comp.add_argument('-m','--inte-method', type=str, default = 'inte', 
                             choices=['inte', 'mbar'], 
                             help='the method of thermodynamic integration')
    parser_comp.add_argument('-s','--scheme', type=str, default = 'simpson', 
                             help='the numeric integration scheme')
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

    args = parser.parse_args()
    return exec_args(args=args, parser=None)

def exec_args(args, parser):
    if args.command is None :
        parser.print_help()
        exit
    if args.command == 'gen' :
        output = args.output
        with open(args.PARAM, 'r') as j:
            jdata = json.load(j)
        make_tasks(output, jdata)
    if args.command == 'refine' :
        refine_tasks(args.input, args.output, args.error)
    elif args.command == 'compute' :
        with open(os.path.join(args.JOB, 'conf.lmp'), 'r') as conf_lmp:
            # fp_conf = open(os.path.join(args.JOB, 'conf.lmp'))
            sys_data = lmp.to_system_data(conf_lmp.read().split('\n'))
        natoms = sum(sys_data['atom_numbs'])
        with open(os.path.join(args.JOB, 'in.json'), 'r') as j:
            jdata = json.load(j)

        if 'copies' in jdata :
            natoms *= np.prod(jdata['copies'])
        nmols = natoms // 3
        fe, fe_err, thermo_info = post_tasks(args.JOB, nmols, method = args.inte_method)
        _print_thermo_info(thermo_info)
        print ('# numb atoms: %d' % natoms)
        print ('# numb  mols: %d' % nmols)        
        print_format = '%20.12f  %10.3e  %10.3e'
        # if args.type == 'helmholtz' :
        print('# Helmholtz free ener per mol (err) [eV]:')
        print(print_format % (fe, fe_err[0], fe_err[1]))
        if args.type == 'gibbs':
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
            e1 = fe + pv
            e1_err = np.sqrt(fe_err[0]**2 + pv_err**2)
            print('# Gibbs free ener per mol (err) [eV]:')
            print(print_format % (e1, e1_err, fe_err[1]))
    
if __name__ == '__main__' :
    _main()
