#!/usr/bin/env python3

import os, sys, json, argparse, glob, shutil
import numpy as np
import scipy.constants as pc
import pymbar

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from dpti.lib.utils import create_path, relative_link_file
from dpti.lib.utils import get_first_matched_key_from_dict
from dpti.lib.utils import copy_file_list
from dpti.lib.utils import block_avg
from dpti.lib.utils import integrate_range
# from lib.utils import integrate_sys_err
from dpti.lib.utils import compute_nrefine
from dpti.lib.utils import parse_seq, link_file_in_dict
from dpti.lib.utils import get_task_file_abspath
from dpti.lib.lammps import get_thermo
from dpti.lib.lammps import get_natoms

# from dpti.equi import gen_equi_lammps_input

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
    return inv_grid

def _gen_lammps_input (conf_file, 
                       mass_map,
                       model,
                       nsteps,
                       timestep,
                       ens,
                       temp,
                       pres=1.0, 
                       tau_t=0.1,
                       tau_p=0.5,
                       thermo_freq=100,
                       copies=None,
                       if_meam=False,
                       meam_model=None):
    ret = ''
    ret += 'clear\n'
    ret += '# --------------------- VARIABLES-------------------------\n'
    ret += 'variable        NSTEPS          equal %d\n' % nsteps
    ret += 'variable        THERMO_FREQ     equal %d\n' % thermo_freq
    # ret += 'variable        DUMP_FREQ       equal %d\n' % thermo_freq
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
    # if if_meam:
    #     ret += 'pair_style      meam \n'
    #     ret += 'pair_coeff      * * /home/fengbo/4_Sn/meam_files/library_18Metal.meam Sn /home/fengbo/4_Sn/meam_files/Sn_18Metal.meam Sn\n'
    if if_meam:
        ret += 'pair_style      meam\n'
        ret += f'pair_coeff      * * {meam_model["library"]} {meam_model["element"]} {meam_model["potential"]} {meam_model["element"]}\n'
    else:
        ret += 'pair_style      deepmd %s\n' % model
        ret += 'pair_coeff\n'
    ret += '# --------------------- MD SETTINGS ----------------------\n'    
    ret += 'neighbor        1.0 bin\n'
    ret += 'timestep        %s\n' % timestep
    ret += 'thermo          ${THERMO_FREQ}\n'
    ret += 'compute         allmsd all msd\n'
    if ens == 'nvt' :        
        ret += 'thermo_style    custom step ke pe etotal enthalpy temp press vol c_allmsd[*]\n'
    elif 'npt' in ens :
        ret += 'thermo_style    custom step ke pe etotal enthalpy temp press vol c_allmsd[*]\n'
    else :
        raise RuntimeError('unknow ensemble %s\n' % ens)                
    ret += '# dump            1 all custom ${DUMP_FREQ} traj.dump id type x y z\n'
    if ens == 'nvt' :
        ret += 'fix             1 all nvt temp ${TEMP} ${TEMP} ${TAU_T}\n'
    elif ens == 'npt-iso' or ens == 'npt':
        ret += 'fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}\n'
    elif ens == 'npt-aniso' :
        ret += 'fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} aniso ${PRES} ${PRES} ${TAU_P}\n'
    elif ens == 'npt-tri' :
        ret += 'fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} tri ${PRES} ${PRES} ${TAU_P}\n'
    elif ens == 'npt-xy' :
        ret += 'fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} aniso ${PRES} ${PRES} ${TAU_P} couple xy\n'
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


def make_tasks(iter_name, jdata, if_meam=None):
    ti_settings = jdata.copy()
    if if_meam is None:
        if_meam = jdata.get('if_meam', None)
    equi_conf = jdata['equi_conf']
    equi_conf = os.path.abspath(equi_conf)
    copies = None
    if 'copies' in jdata :
        copies = jdata['copies']
    model = jdata['model']
    meam_model = jdata.get('meam_model', None)
    # model = os.path.abspath(model)
    # mass_map = jdata['mass_map']
    mass_map = get_first_matched_key_from_dict(jdata, ['model_mass_map', 'mass_map'])
    nsteps = jdata['nsteps']
    # timestep = jdata['timestep']
    timestep = get_first_matched_key_from_dict(jdata, ['timestep', 'dt'])
    # thermo_freq = jdata['thermo_freq']
    thermo_freq = get_first_matched_key_from_dict(jdata, ['thermo_freq', 'stat_freq'])
    # thermos = jdata['thermos']
    ens = jdata['ens']
    path = jdata['path']
    if 'nvt' in ens :
        if path == 't' :
            temp_seq = get_first_matched_key_from_dict(jdata, ['temp_seq', 'temps'])
            temp_list = parse_seq(temp_seq)
            tau_t = jdata['tau_t']
            ntasks = len(temp_list)
        else :
            raise RuntimeError('supported path of nvt ens is \'t\'')
    elif 'npt' in ens :
        if path == 't' :
            temp_seq = get_first_matched_key_from_dict(jdata, ['temp_seq', 'temps'])
            temp_list = parse_seq(temp_seq)
            pres = get_first_matched_key_from_dict(jdata, ['pres', 'press'])
            ntasks = len(temp_list)
        elif path == 't-ginv' :
            temp_seq = get_first_matched_key_from_dict(jdata, ['temp_seq', 'temps'])
            temp_list = parse_seq_ginv(temp_seq)
            pres = get_first_matched_key_from_dict(jdata, ['pres', 'press'])
            ntasks = len(temp_list)
        elif path == 'p' :
            temp = get_first_matched_key_from_dict(jdata, ['temp', 'temps'])
            pres_seq = get_first_matched_key_from_dict(jdata, ['pres_seq', 'press'])
            pres_list = parse_seq(pres_seq)
            ntasks = len(pres_list)
        else :
            raise RuntimeError('supported path of npt ens are \'t\' or \'p\'')
        tau_t = jdata['tau_t']
        tau_p = jdata['tau_p']
    else :
        raise RuntimeError('invalid ens')

    job_abs_dir = create_path(iter_name)




    # dct1 = link_file_in_dict(
    #     dct=jdata,
    #     key_list=["equi_conf", "model"],
    #     target_dir=job_abs_dir
    # )
    # ti_settings.update(dct1)

    # meam_model = jdata.get('meam_model', None)
    # dct2 = link_file_in_dict(
    #     dct=meam_model,
    #     key_list=["library", "potential"],
    #     target_dir=job_abs_dir
    # )
    # if meam_model:
    #     ti_settings['meam_model'].update(dct2)

    # link_file_dict = {}
    # link_file_dict.update(dct1)
    # link_file_dict.update(dct2)


    # copied_conf = os.path.join(os.path.abspath(iter_name), 'conf.lmp')
    # shutil.copyfile(equi_conf, copied_conf)
    # jdata['equi_conf'] = 'conf.lmp'
    # linked_model = os.path.join(os.path.abspath(iter_name), 'graph.pb')
    # shutil.copyfile(model, linked_model)
    # jdata['model'] = 'graph.pb'

    # cwd = os.getcwd()
    # os.chdir(iter_name)
    relative_link_file(equi_conf, job_abs_dir)
    if model:
        relative_link_file(model, job_abs_dir)
    if if_meam:
        relative_link_file(meam_model['library'], job_abs_dir)
        relative_link_file(meam_model['potential'], job_abs_dir)

    with open(os.path.join(job_abs_dir, 
        'ti_settings.json'), 'w') as fp:
        json.dump(ti_settings, fp, indent=4)

    with open(os.path.join(job_abs_dir, 'ti_settings.json'),'w') as f:
            json.dump(ti_settings, f, indent=4)

    for ii in range(ntasks) :
        task_dir = os.path.join(job_abs_dir, 'task.%06d' % ii)
        task_abs_dir = create_path(task_dir)
        # os.chdir(work_path)

        relative_link_file(equi_conf, task_abs_dir)
        if model:
            relative_link_file(model, task_abs_dir)
        if if_meam:
            relative_link_file(meam_model['library'], task_abs_dir)
            relative_link_file(meam_model['potential'], task_abs_dir)

        # for file in list(link_file_dict.values()):
        #     file_path = os.path.join(job_abs_dir, file)
        #     relative_link_file(file_path, task_abs_dir)
            # os.symlink()
        # os.symlink(os.path.relpath(copied_conf), 'conf.lmp')
        # os.symlink(os.path.relpath(linked_model), 'graph.pb')
        if 'nvt' in ens and path == 't' :
            lmp_str \
                = _gen_lammps_input(os.path.basename(equi_conf),
                                    mass_map, 
                                    model,
                                    nsteps, 
                                    timestep,
                                    ens,
                                    temp_list[ii],
                                    pres=pres,
                                    tau_t = tau_t,
                                    thermo_freq = thermo_freq, 
                                    copies = copies,
                                    if_meam=if_meam,
                                    meam_model=meam_model
                                    )
            thermo_out = temp_list[ii]
            # with open('thermo.out', 'w') as fp :
            #     fp.write('%f' % temps[ii])
        elif 'npt' in ens and (path == 't' or path == 't-ginv'):
            lmp_str \
                = _gen_lammps_input(os.path.basename(equi_conf),
                                    mass_map, 
                                    model,
                                    nsteps, 
                                    timestep,
                                    ens,
                                    temp_list[ii],
                                    pres,
                                    tau_t = tau_t,
                                    tau_p = tau_p,
                                    thermo_freq = thermo_freq, 
                                    copies = copies,
                                    if_meam=if_meam,
                                    meam_model=meam_model
                                    )
            thermo_out = temp_list[ii]
            # with open('thermo.out', 'w') as fp :
            #     fp.write('%f' % (temps[ii]))
        elif 'npt' in ens and path == 'p' :
            lmp_str \
                = _gen_lammps_input(os.path.basename(equi_conf),
                                    mass_map, 
                                    model,
                                    nsteps, 
                                    timestep,
                                    ens,
                                    temp,
                                    pres_list[ii],
                                    tau_t = tau_t,
                                    tau_p = tau_p,
                                    thermo_freq = thermo_freq, 
                                    copies = copies,
                                    if_meam=if_meam,
                                    meam_model=meam_model
                                )
            thermo_out = pres_list[ii]
        else:
            raise RuntimeError('invalid ens or path setting' )
        
        with open(os.path.join(task_abs_dir, 'thermo.out'), 'w') as fp:
            fp.write('%f' % (thermo_out))
        with open(os.path.join(task_abs_dir, 'in.lammps'), 'w') as fp:
            fp.write(lmp_str)
        
        # os.chdir(cwd)

def _compute_thermo (lmplog, natoms, stat_skip, stat_bsize) :
    data = get_thermo(lmplog)
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

def _print_thermo_info(info, more_head = '') :
    ptr = '# thermodynamics (normalized by natoms) %s\n' % more_head
    ptr += '# E (err)  [eV]:  %20.8f %20.8f\n' % (info['e'], info['e_err'])
    ptr += '# H (err)  [eV]:  %20.8f %20.8f\n' % (info['h'], info['h_err'])
    ptr += '# T (err)   [K]:  %20.8f %20.8f\n' % (info['t'], info['t_err'])
    ptr += '# P (err) [bar]:  %20.8f %20.8f\n' % (info['p'], info['p_err'])
    ptr += '# V (err) [A^3]:  %20.8f %20.8f\n' % (info['v'], info['v_err'])
    ptr += '# PV(err)  [eV]:  %20.8f %20.8f' % (info['pv'], info['pv_err'])
    print(ptr)

def _thermo_inte(jdata, Eo, Eo_err, all_t, integrand, integrand_err, scheme = 's', all_e=None) :
    path = jdata['path']
    ens = jdata['ens']
    all_temps = []
    all_press = []
    all_fe = []
    all_fe_err = []
    all_fe_sys_err = []
    
    # if all_e is not None:
    #     print('switch to test integral mode by yuanfengbo')
    #     array_e = np.asarray(all_e, dtype='float64')
    #     array_t = np.asarray(all_t, dtype='float64')
    #     array_t_delta = array_t[1:] - array_t[:-1]
    #     array_e_delta = array_e[1:] - array_e[:-1]
    #     array_b = array_e_delta / array_t_delta
    #     array_a = array_e[:-1] - array_t[:-1] * array_b 
    #     array_diff_e = array_a * (array_t[1:] - array_t[:-1])/(array_t[1:]*array_t[:-1]) + array_b * np.log(array_t[1:]/array_t[:-1])
        
    #     print('!!!', array_a, array_b, all_e , array_t_delta, array_e_delta, array_diff_e)
        
    #     # for ii in range(0, len(array_t)):
    #     for ii in range(len(array_t)-1, -1, -1):
    #         e1 = (Eo / (array_t[-1]) + np.sum(array_diff_e[ii:])) * array_t[ii]
    #         # e1 = (Eo / (array_t[0]) - np.sum(array_diff_e[0:ii])) * array_t[ii]
    #         all_temps.append(array_t[ii])
    #         err = 0
    #         sys_err = 0
    #         all_press.append(jdata['pres'])
    #         all_fe.append(e1)
    #         all_fe_err.append(err)
    #         all_fe_sys_err.append(sys_err)

    #     return np.asarray(all_temps), np.asarray(all_press), np.asarray(all_fe), np.asarray(all_fe_err), np.asarray(all_fe_sys_err)
            
            
        
    all_t, inte, inte_e, stat_e = integrate_range(all_t, integrand, integrand_err, scheme)
    for ii in range(0, len(all_t)):
        diff_e = inte[ii]
        err = stat_e[ii]
        sys_err = inte_e[ii]
        # diff_e, err = integrate(all_t[0:ii+1], integrand[0:ii+1], integrand_err[0:ii+1], scheme)
        # sys_err = integrate_sys_err(all_t[0:ii+1], integrand[0:ii+1], scheme)
        if path == 't' or path == 't-ginv':
            e1 = (Eo / (all_t[0]) - diff_e) * all_t[ii]
            err = np.sqrt(np.square(Eo_err / all_t[0]) + np.square(err))
            err *= all_t[ii]
            sys_err *= all_t[ii]
            all_temps.append(all_t[ii])
            if 'npt' in ens :
                all_press.append(jdata['pres'])
        elif path == 'p':
            e1 = Eo + diff_e        
            err = np.sqrt(np.square(Eo_err) + np.square(err))
            all_temps.append(jdata['temp'])
            all_press.append(all_t[ii])
        all_fe.append(e1)
        all_fe_err.append(err)
        all_fe_sys_err.append(sys_err)
    return np.asarray(all_temps), np.asarray(all_press), np.asarray(all_fe), np.asarray(all_fe_err), np.asarray(all_fe_sys_err)

def post_tasks(iter_name, jdata, Eo, Eo_err = 0, To = None, natoms = None, scheme = 'simpson', shift = 0.0) :
    equi_conf = get_task_file_abspath(iter_name, jdata['equi_conf'])
    if natoms == None :        
        natoms = get_natoms(equi_conf)
        if 'copies' in jdata :
            natoms *= np.prod(jdata['copies'])
    stat_skip = jdata['stat_skip']
    stat_bsize = jdata['stat_bsize']
    ens = jdata['ens']
    path = jdata['path']

    all_tasks = glob.glob(os.path.join(iter_name, 'task.[0-9]*'))
    all_tasks.sort()
    ntasks = len(all_tasks)
    
    all_t = []
    all_e = []
    all_e_err = []
    integrand = []
    integrand_err = []
    all_enthalpy = []
    all_msd_xyz = []
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
    print('# natoms: %d' % natoms)

    for ii in all_tasks :
        # get T or P
        thermo_name = os.path.join(ii, 'thermo.out')
        tt = float(open(thermo_name).read())
        all_t.append(tt)
        # get energy stat
        log_name = os.path.join(ii, 'log.lammps')
        data = get_thermo(log_name)
        np.savetxt(os.path.join(ii, 'data'), data, fmt = '%.6e')
        ea, ee = block_avg(data[:, stat_col], 
                           skip = stat_skip, 
                           block_size = stat_bsize)
        enthalpy, _ = block_avg(data[:, 5], skip = stat_skip, block_size = stat_bsize)
        msd_xyz = data[-1, -1]
        # COM corr
        if path == 't' or path == 't-ginv' :
            ea += 1.5 * pc.Boltzmann * tt / pc.electron_volt
            # print('~~', tt, ea, 1.5 * pc.Boltzmann * tt / pc.electron_volt)
        elif path == 'p' :
            temp = jdata['temp']
            ea += 1.5 * pc.Boltzmann * temp / pc.electron_volt
        else :
            raise RuntimeError('invalid path setting' )
        # normalized by number of atoms
        ea /= natoms
        if path == 't' or path == 't-ginv':
            ea -= shift
        ee /= np.sqrt(natoms)
        all_e.append(ea)
        all_e_err.append(ee)
        all_enthalpy.append(enthalpy)
        all_msd_xyz.append(msd_xyz)
        # gen integrand
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
    all_print.append(all_t)
    all_print.append(integrand)
    all_print.append(all_e)
    all_print.append(all_e_err)
    all_print.append(all_enthalpy)
    all_print.append(all_msd_xyz)
    all_print = np.array(all_print)
    np.savetxt(os.path.join(iter_name, 'ti.out'), 
               all_print.T, 
               fmt = '%.8e', 
               header = 't/p Integrand U/V U/V_err enthalpy msd_xyz')

    info0 = _compute_thermo(os.path.join(all_tasks[ 0], 'log.lammps'), natoms, stat_skip, stat_bsize)
    info1 = _compute_thermo(os.path.join(all_tasks[-1], 'log.lammps'), natoms, stat_skip, stat_bsize)
    _print_thermo_info(info0, 'at start point')
    _print_thermo_info(info1, 'at end point')

    if To is not None :
        index = all_t.index(To)
        if index == None :
            if 'nvt' == ens :
                raise RuntimeError('cannot find %f in T', To)
            elif 'npt' in ens :
                raise RuntimeError('cannot find %f in P', To)
        all_t_1 = all_t[0:index+1]
        integrand_1 = integrand[0:index+1]
        integrand_err_1 = integrand_err[0:index+1]
        all_t_1 = np.flip(all_t_1, 0)
        integrand_1 = np.flip(integrand_1, 0)
        integrand_err_1 = np.flip(integrand_err_1, 0)
        all_t_2 = all_t[index:]
        integrand_2 = integrand[index:]
        integrand_err_2 = integrand_err[index:]
        all_temps_1, all_press_1, all_fe_1, all_fe_err_1, all_fe_sys_err_1 \
            = _thermo_inte(jdata, Eo, Eo_err, all_t_1, integrand_1, integrand_err_1, scheme = scheme)
        all_temps_2, all_press_2, all_fe_2, all_fe_err_2, all_fe_sys_err_2 \
            = _thermo_inte(jdata, Eo, Eo_err, all_t_2, integrand_2, integrand_err_2, scheme = scheme)
        all_temps_1 = np.flip(all_temps_1, 0)
        all_press_1 = np.flip(all_press_1, 0)
        all_fe_1 = np.flip(all_fe_1, 0)
        all_fe_err_1 = np.flip(all_fe_err_1, 0)
        all_fe_sys_err_1 = np.flip(all_fe_sys_err_1, 0)
        all_temps = np.append(all_temps_1, all_temps_2[1:])
        all_press = np.append(all_press_1, all_press_2[1:])
        all_fe = np.append(all_fe_1, all_fe_2[1:])
        all_fe_err = np.append(all_fe_err_1, all_fe_err_2[1:])
        all_fe_sys_err = np.append(all_fe_sys_err_1, all_fe_sys_err_2[1:])
    else :    
        all_temps, all_press, all_fe, all_fe_err, all_fe_sys_err \
            = _thermo_inte(jdata, Eo, Eo_err, all_t, integrand, integrand_err, scheme = scheme, all_e=all_e)

    # print('ti.py:debug:data', data)
    result = ""
    # result_file = open(f"{iter_name}/../result", 'w')
    if 'nvt' == ens :
        print('#%8s  %20s  %9s  %9s  %9s' % ('T(ctrl)', 'F', 'stat_err', 'inte_err', 'err'))
        result += ('#%8s  %20s  %9s  %9s  %9s\n' % ('T(ctrl)', 'F', 'stat_err', 'inte_err', 'err'))
        for ii in range(len(all_temps)) :
            print ('%9.2f  %20.12f  %9.2e  %9.2e  %9.2e' 
                   % (all_temps[ii], all_fe[ii], all_fe_err[ii], all_fe_sys_err[ii], np.linalg.norm([all_fe_err[ii], all_fe_sys_err[ii]])))
            result += ('%9.2f  %20.12f  %9.2e  %9.2e  %9.2e\n' 
                   % (all_temps[ii], all_fe[ii], all_fe_err[ii], all_fe_sys_err[ii], np.linalg.norm([all_fe_err[ii], all_fe_sys_err[ii]])))
    elif 'npt' in ens :
        print('#%8s  %15s  %20s  %9s  %9s  %9s' % ('T(ctrl)', 'P(ctrl)', 'F', 'stat_err', 'inte_err', 'err'))
        result += ('#%8s  %15s  %20s  %9s  %9s  %9s\n' % ('T(ctrl)', 'P(ctrl)', 'F', 'stat_err', 'inte_err', 'err'))
        for ii in range(len(all_temps)) :
            print ('%9.2f  %15.8e  %20.12f  %9.2e  %9.2e  %9.2e' 
                   % (all_temps[ii], all_press[ii], all_fe[ii], all_fe_err[ii], all_fe_sys_err[ii], np.linalg.norm([all_fe_err[ii], all_fe_sys_err[ii]])))
            result += ('%9.2f  %15.8e  %20.12f  %9.2e  %9.2e  %9.2e\n'
                   % (all_temps[ii], all_press[ii], all_fe[ii], all_fe_err[ii], all_fe_sys_err[ii], np.linalg.norm([all_fe_err[ii], all_fe_sys_err[ii]])))
            # print(all_temps[ii], all_press[ii], all_fe[ii], all_fe_err[ii], all_fe_sys_err[ii], np.linalg.norm([all_fe_err[ii], all_fe_sys_err[ii]]))
    # result_file.close()

    data = dict(all_temps=all_temps.tolist(), all_press=all_press.tolist(),
        all_fe=all_fe.tolist(), all_fe_stat_err=all_fe_err.tolist(), all_fe_inte_err=all_fe_sys_err.tolist(), 
        all_fe_tot_err=np.linalg.norm([all_fe_err[ii], all_fe_sys_err[ii]]).tolist())

    # data = [all_temps.tolist(), all_press.tolist(), 
    #     all_fe.tolist(), all_fe_err.tolist(), all_fe_sys_err.tolist(), 
    #     np.linalg.norm([all_fe_err[ii], all_fe_sys_err[ii]]).tolist()]
    info = dict(start_point_info=info0, end_point_info=info1, data=data)
    # print('result', result)
    with open(os.path.join(iter_name, '../', 'result'), 'w') as f:
        f.write(result)
    with open(os.path.join(iter_name, 'result.json'), 'w') as f:
        f.write(json.dumps(info))
    return info


def post_tasks_mbar(iter_name, jdata, Eo, natoms = None) :
    equi_conf = jdata['equi_conf']
    if natoms == None :        
        natoms = get_natoms(equi_conf)
        if 'copies' in jdata :
            natoms *= np.prod(jdata['copies'])
    stat_skip = jdata['stat_skip']
    stat_bsize = jdata['stat_bsize']
    ens = jdata['ens']
    path = jdata['path']

    all_tasks = glob.glob(os.path.join(iter_name, 'task.[0-9]*'))
    all_tasks.sort()
    ntasks = len(all_tasks)

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
    print('# natoms: %d' % natoms)

    all_t = []
    for ii in all_tasks :
        thermo_name = os.path.join(ii, 'thermo.out')
        tt = float(open(thermo_name).read())
        all_t.append(tt)
    all_t = np.array(all_t)
    nt = all_t.size
    
    ukn = None
    nk = []    
    for ii in all_tasks :
        log_name = os.path.join(ii, 'log.lammps')
        data = get_thermo(log_name)
        np.savetxt(os.path.join(ii, 'data'), data, fmt = '%.6e')
        block_u = []
        if path == 't' or path == 't-ginv':
            this_e = data[stat_skip::1, stat_col]
            nk.append(this_e.size)
            for tt in all_t :
                kt_in_ev = pc.Boltzmann * tt / pc.electron_volt
                block_u.append(this_e / kt_in_ev)
        elif path == 'p' :
            this_e = data[stat_skip::1, 3]
            this_v = data[stat_skip::1, 7]
            nk.append(this_e.size)            
            # # cvt from barA^3 to eV
            unit_cvt = 1e5 * (1e-10**3) / pc.electron_volt
            temp = jdata['temps']
            kt_in_ev = temp * pc.Boltzmann / pc.electron_volt
            for tt in all_t :
                block_u.append((this_e + tt * this_v * unit_cvt) / kt_in_ev)
        else:
            raise RuntimeError('invalid path setting' )
        block_u = np.reshape(block_u, [nt, -1])
        if ukn is None :
            ukn = block_u 
        else :
            ukn = np.concatenate((ukn, block_u), axis = 1)
    nk = np.array(nk)

    info0 = _compute_thermo(os.path.join(all_tasks[ 0], 'log.lammps'), natoms, stat_skip, stat_bsize)
    info1 = _compute_thermo(os.path.join(all_tasks[-1], 'log.lammps'), natoms, stat_skip, stat_bsize)
    _print_thermo_info(info0, 'at start point')
    _print_thermo_info(info1, 'at end point')

    mbar = pymbar.MBAR(ukn, nk)
    Deltaf_ij, dDeltaf_ij, Theta_ij = mbar.getFreeEnergyDifferences()
    Deltaf_ij = Deltaf_ij / natoms
    dDeltaf_ij = dDeltaf_ij / np.sqrt(natoms)

    all_temps = []
    all_press = []
    all_fe = []
    all_fe_err = []
    all_fe_sys_err = []
    for ii in range(0, nt) :
        if path == 't' or path == 't-ginv':
            kt_in_ev = all_t[ii] * pc.Boltzmann / pc.electron_volt
            e1 = (Eo / (all_t[0])) * all_t[ii] + Deltaf_ij[0,ii] * kt_in_ev
            err = dDeltaf_ij[0,ii] * kt_in_ev
            sys_err = 0
            all_temps.append(all_t[ii])
            if 'npt' in ens :
                all_press.append(jdata['pres'])
        elif path == 'p':
            kt_in_ev = jdata['temps'] * pc.Boltzmann / pc.electron_volt
            e1 = Eo + Deltaf_ij[0,ii] * kt_in_ev
            err = dDeltaf_ij[0,ii] * kt_in_ev
            sys_err = 0
            all_temps.append(jdata['temp'])
            all_press.append(all_t[ii])            
        else :
            pass
        all_fe.append(e1)
        all_fe_err.append(err)
        all_fe_sys_err.append(sys_err)

    if 'nvt' == ens :
        print('#%8s  %15s  %9s  %9s' % ('T(ctrl)', 'F', 'stat_err', 'inte_err'))
        for ii in range(len(all_temps)) :
            print ('%9.2f  %20.12f  %9.2e  %9.2e' 
                   % (all_temps[ii], all_fe[ii], all_fe_err[ii], all_fe_sys_err[ii]))
    elif 'npt' in ens :
        print('#%8s  %15s  %15s  %9s  %9s' % ('T(ctrl)', 'P(ctrl)', 'F', 'stat_err', 'inte_err'))
        for ii in range(len(all_temps)) :
            print ('%9.2f  %15.8e  %20.12f  %9.2e  %9.2e' 
                   % (all_temps[ii], all_press[ii], all_fe[ii], all_fe_err[ii], all_fe_sys_err[ii]))
    # info = dict(start_point_info=info0, end_point_info=info1, all_temps=list(all_temps), all_press=list(all_press),
    #              all_fe=list(all_fe), all_fe_err=list(all_fe_err), all_fe_sys_err=list(all_fe_sys_err))
    # open(os.path.join(iter_name, 'result.json'), 'w').write(json.dumps(info))
    # return info


def refine_task (from_task, to_task, err) :
    from_task = os.path.abspath(from_task)
    to_task = os.path.abspath(to_task)
    from_json = os.path.join(from_task, 'in.json')
    to_json = os.path.join(to_task, 'in.json')
    from_jdata = json.load(open(from_json))
    to_jdata = from_jdata
    path = from_jdata['path']

    from_ti = os.path.join(from_task, 'ti.out')
    if not os.path.isfile(from_ti) :
        raise RuntimeError("cannot find file %s, task should be computed befor refined" % from_ti)
    tmp_array = np.loadtxt(from_ti)
    all_t = tmp_array[:,0]
    integrand = tmp_array[:,1]
    ntask = all_t.size

    if path == 't' or path == 't-ginv':
        interval_nrefine = compute_nrefine(all_t, integrand, err, all_t)
    elif path == 'p' :
        interval_nrefine = compute_nrefine(all_t, integrand, err)
    else :
        raise RuntimeError('unknow path ' + path)

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
    
    if to_jdata['path'] == 't-ginv' :
        to_jdata['path'] = 't'        
    if to_jdata['path'] == 't' :
        to_jdata['temps'] = refined_t
    elif to_jdata['path'] == 'p' :
        to_jdata['press'] = refined_t
    else :
        raise RuntimeError('unknow path ' + path)
    to_jdata['orig_task'] = from_task
    to_jdata['back_map'] = back_map
    to_jdata['refine_error'] = err
    to_jdata['equi_conf'] = get_task_file_abspath(from_task, from_jdata['equi_conf'])
    to_jdata['model'] = get_task_file_abspath(from_task, from_jdata['model'])
    # create_path(to_task)
    # with open(to_json, 'w') as fp :
    #     json.dump(to_jdata, fp, indent=4)

    make_tasks(to_task, to_jdata)
    
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
            
def compute_task(job, inte_method, Eo, Eo_err, To, scheme='simpson'):
    # job = args.JOB
    with open(os.path.join(job, 'ti_settings.json'), 'r') as f:
        jdata = json.load(f)
    if inte_method == 'inte' :
        info = post_tasks(job, jdata, Eo=Eo, Eo_err=Eo_err, To=To, scheme=scheme)
    elif inte_method == 'mbar' :
        info = post_tasks_mbar(job, jdata, Eo)
    else :
        raise RuntimeError('unknow integration method')
    return info
    

def _main ():
    parser = argparse.ArgumentParser(
        description="Compute free energy by TI")
    subparsers = parser.add_subparsers(title='Valid subcommands', dest='command', help = 'valid commands')

    parser_gen = subparsers.add_parser('gen', help='Generate a job')
    parser_gen.add_argument('PARAM', type=str ,
                            help='json parameter file')
    parser_gen.add_argument('-o','--output', type=str, default = 'new_job',
                            help='the output folder for the job')
    parser_gen.add_argument("-z", "--meam", help="whether use meam instead of dp", action="store_true")

    parser_comp = subparsers.add_parser('compute', help= 'Compute the result of a job')
    parser_comp.add_argument('JOB', type=str ,
                             help='folder of the job')
    parser_comp.add_argument('-m','--inte-method', type=str, default = 'inte', 
                             choices=['inte', 'mbar'], 
                             help='the method of thermodynamic integration')
    parser_comp.add_argument('-e', '--Eo', type=float, default = 0,
                             help='free energy of starting point')
    parser_comp.add_argument('-E', '--Eo-err', type=float, default = 0,
                             help='The statistical error of the starting free energy')
    parser_comp.add_argument('-t', '--To', type=float, 
                             help='the starting thermodynamic position')
    parser_comp.add_argument('-s', '--scheme', type=str, default = 'simpson',
                             help='the numerical integration scheme')

    parser_comp = subparsers.add_parser('refine', help= 'Refine the grid of a job')
    parser_comp.add_argument('-i', '--input', type=str, required=True,
                             help='input job')
    parser_comp.add_argument('-o', '--output', type=str, required=True,
                             help='output job')
    parser_comp.add_argument('-e', '--error', type=float, required=True,
                             help='the error required')
    args = parser.parse_args()

    if args.command is None :
        parser.print_help()
        exit
    if args.command == 'gen' :
        output = args.output
        jdata = json.load(open(args.PARAM, 'r'))
        make_tasks(output, jdata, if_meam=args.meam)
    elif args.command == 'compute' :
        compute_task(args.JOB, inte_method=args.inte_method, Eo=args.Eo, Eo_err=args.Eo_err, To=args.To, scheme=args.scheme)
    #     job = args.JOB
    #     jdata = json.load(open(os.path.join(job, 'in.json'), 'r'))
    #     if args.inte_method == 'inte' :
    #         post_tasks(job, jdata, args.Eo, Eo_err = args.Eo_err, To = args.To, scheme = args.scheme)
    #     elif args.inte_method == 'mbar' :
    #         post_tasks_mbar(job, jdata, args.Eo)
    #     else :
    #         raise RuntimeError('unknow integration method')
    elif args.command == 'refine' :
        refine_task(args.input, args.output, args.error)

    
if __name__ == '__main__' :
    _main()
        
