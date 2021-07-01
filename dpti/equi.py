#!/usr/bin/env python3

import os, sys, json, argparse, glob
import numpy as np
import scipy.constants as pc

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
import dpti
from dpti.lib.utils import create_dict_not_empty_key, create_path, relative_link_file
from dpti.lib.utils import block_avg, link_file_in_dict
from dpti.lib.water import compute_bonds
from dpti.lib.water import posi_diff
from dpti.lib.utils import get_task_file_abspath
# import dpti

from dpti.lib.lammps import get_natoms, get_last_dump, get_thermo
from dpti.lib.lmp import from_system_data
from dpti.lib.dump import system_data

from dargs import dargs, Argument, Variant
# from lib import dump 
# from .lib import lammps
# from .lib import dump
# from .lib import lmp

# def _gen_lammps_input
# def gen_equi_header(nsteps, prt_freq, dump_freq, temp, pres, tau_t, tau_p, mass_map, conf_file):
# def gen_equi_header(nsteps, prt_freq, dump_freq, temp, pres, tau_t, tau_p, mass_map, conf_file):
def gen_equi_header(nsteps, 
        thermo_freq, 
        dump_freq,
        mass_map,
        temp,
        tau_t,
        tau_p,
        equi_conf,
        pres=None
    ):
    ret = ''
    ret += 'clear\n'
    ret += '# --------------------- VARIABLES-------------------------\n'
    ret += 'variable        NSTEPS          equal %d\n' % nsteps
    ret += 'variable        THERMO_FREQ     equal %d\n' % thermo_freq
    ret += 'variable        DUMP_FREQ       equal %d\n' % dump_freq
    ret += 'variable        NREPEAT         equal ${NSTEPS}/${DUMP_FREQ}\n'
    ret += 'variable        TEMP            equal %.6f\n' % temp
    # if equi_settings['pres'] is not None :
    if pres is not None:
        ret += 'variable        PRES            equal %.6f\n' % pres
    ret += 'variable        TAU_T           equal %.6f\n' % tau_t
    ret += 'variable        TAU_P           equal %.6f\n' % tau_p
    ret += '# ---------------------- INITIALIZAITION ------------------\n'
    ret += 'units           metal\n'
    ret += 'boundary        p p p\n'
    ret += 'atom_style      atomic\n'
    ret += '# --------------------- ATOM DEFINITION ------------------\n'
    ret += 'box             tilt large\n'
    ret += 'read_data       %s\n' % equi_conf
    ret += 'change_box      all triclinic\n'
    for jj in range(len(mass_map)):
        ret+= "mass            %d %.6f\n" %(jj+1, mass_map[jj])
    return ret
    
# def gen_equi_force_field(model, if_meam=None):
def gen_equi_force_field(model, if_meam=False, meam_model=None):
    # equi_settings =
    # model = equi_settings['model']
    # assert type(model) is dict, f"equi_settings['model] must be a dict. model:{model}"
    # model_type = equi_settings.get('model_type', None)
    # if model_type == 'deepmd':
    #     assert 'deepmd_model' in equi_settings, f" 'deepmd_model' must be in equi_settings. {equi_settings}"
    # elif model_type == 'meam':
    #     assert 'meam_library' in equi_settings, f" 'meam_library' must be in equi_settings. {equi_settings}"
    #     assert 'meam_potential' in equi_settings, f" 'meam_potential' must be in equi_settings. {equi_settings}"
    #     assert 'meam_element' in equi_settings, f" 'meam_element' must be in equi_settings. {equi_settings}"
    # else: 
    #     raise ValueError(f" model_type:{model_type} must be in ['deepmd', 'meam'];"
    #         f"equi_settings:{equi_settings}")

    ret = ''
    ret += '# --------------------- FORCE FIELDS ---------------------\n'
    if not if_meam:
        ret += 'pair_style      deepmd %s\n' % model
        ret += 'pair_coeff\n'
    else:
        meam_library = meam_model['library']
        meam_potential = meam_model['potential']
        meam_element = meam_model['element']
        ret += 'pair_style      meam\n'
        ret += 'pair_coeff      * * %s %s %s %s\n' % (meam_library,
            meam_element, meam_potential, meam_element)
    return ret

def gen_equi_thermo_settings(timestep):
    ret = ''
    ret += '# --------------------- MD SETTINGS ----------------------\n'    
    ret += 'neighbor        1.0 bin\n'
    ret += 'timestep        %.6f\n' % timestep
    ret += 'thermo          ${THERMO_FREQ}\n'
    ret += 'compute         allmsd all msd\n'
    ret += 'thermo_style    custom step ke pe etotal enthalpy temp press vol lx ly lz xy xz yz pxx pyy pzz pxy pxz pyz c_allmsd[*]\n'
    return ret

def gen_equi_dump_settings(if_dump_avg_posi):
    ret = ''
    if if_dump_avg_posi:
        ret += 'compute         ru all property/atom xu yu zu\n'
        ret += 'fix             ap all ave/atom ${DUMP_FREQ} ${NREPEAT} ${NSTEPS} c_ru[1] c_ru[2] c_ru[3]\n'
        ret += 'dump            fp all custom ${NSTEPS} dump.avgposi id type f_ap[1] f_ap[2] f_ap[3]\n'
    ret += 'dump            1 all custom ${DUMP_FREQ} dump.equi id type x y z vx vy vz\n'
    return ret

def gen_equi_ensemble_settings(ens):
    # ens = equi_settings['ens']
    ret = ''
    if ens == 'nvt' :
        ret += 'fix             1 all nvt temp ${TEMP} ${TEMP} ${TAU_T}\n'
    elif ens == 'npt-iso' or ens == 'npt':
        ret += 'fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}\n'
    elif ens == 'npt-xy' :
        ret += 'fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} aniso ${PRES} ${PRES} ${TAU_P} couple xy\n'
    elif ens == 'npt-aniso' :
        ret += 'fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} aniso ${PRES} ${PRES} ${TAU_P}\n'
    elif ens == 'npt-tri' :
        ret += 'fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} tri ${PRES} ${PRES} ${TAU_P}\n'
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

def gen_equi_lammps_input(nsteps, 
        thermo_freq, 
        dump_freq,
        mass_map,
        temp,
        tau_t,
        tau_p,
        equi_conf,
        model,
        timestep,
        if_dump_avg_posi,
        ens,
        pres=None,
        if_meam=False, 
        meam_model=None
    ):
    equi_header = gen_equi_header(nsteps, 
        thermo_freq=thermo_freq, 
        dump_freq=dump_freq,
        mass_map=mass_map,
        temp=temp,
        tau_t=tau_t,
        tau_p=tau_p,
        equi_conf=equi_conf,
        pres=pres
    )
    equi_force_field = gen_equi_force_field(
        model, if_meam=if_meam, meam_model=meam_model
    )
    equi_thermo_settings = gen_equi_thermo_settings(
        timestep=timestep
    )
    equi_dump_settings = gen_equi_dump_settings(
        if_dump_avg_posi=if_dump_avg_posi
    )
    equi_ensemble_settings = gen_equi_ensemble_settings(
        ens=ens
    )

    equi_lammps_input = (equi_header + equi_force_field
        + equi_thermo_settings + equi_dump_settings + equi_ensemble_settings)
    return equi_lammps_input
# def 

# def _gen_lammps_input (conf_file, 
#                        mass_map,
#                        model,
#                        nsteps,
#                        dt,
#                        ens,
#                        temp,
#                        pres = 1.0, 
#                        tau_t = 0.1,
#                        tau_p = 0.5,
#                        prt_freq = 100, 
#                        dump_freq = 1000, 
#                        dump_ave_posi = False,
#                        if_meam = False) :
#     ret = ''
#     ret += 'clear\n'
#     ret += '# --------------------- VARIABLES-------------------------\n'
#     ret += 'variable        NSTEPS          equal %d\n' % nsteps
#     ret += 'variable        THERMO_FREQ     equal %d\n' % prt_freq
#     ret += 'variable        DUMP_FREQ       equal %d\n' % dump_freq
#     ret += 'variable        NREPEAT         equal ${NSTEPS}/${DUMP_FREQ}\n'
#     ret += 'variable        TEMP            equal %f\n' % temp
#     if pres is not None :
#         ret += 'variable        PRES            equal %f\n' % pres
#     ret += 'variable        TAU_T           equal %f\n' % tau_t
#     ret += 'variable        TAU_P           equal %f\n' % tau_p
#     ret += '# ---------------------- INITIALIZAITION ------------------\n'
#     ret += 'units           metal\n'
#     ret += 'boundary        p p p\n'
#     ret += 'atom_style      atomic\n'
#     ret += '# --------------------- ATOM DEFINITION ------------------\n'
#     ret += 'box             tilt large\n'
#     ret += 'read_data       %s\n' % conf_file
#     ret += 'change_box      all triclinic\n'
#     for jj in range(len(mass_map)) :
#         ret+= "mass            %d %f\n" %(jj+1, mass_map[jj])
#     ret += '# --------------------- FORCE FIELDS ---------------------\n'
#     if if_meam:
#         ret += 'pair_style      meam\n'
#         ret += 'pair_coeff      * * /home/fengbo/4_Sn/meam_files/library_18Metal.meam Sn /home/fengbo/4_Sn/meam_files/Sn_18Metal.meam Sn\n'
#     else:
#         ret += 'pair_style      deepmd %s\n' % model
#         ret += 'pair_coeff\n'
#     ret += '# --------------------- MD SETTINGS ----------------------\n'    
#     ret += 'neighbor        1.0 bin\n'
#     ret += 'timestep        %s\n' % dt
#     ret += 'thermo          ${THERMO_FREQ}\n'
#     ret += 'compute         allmsd all msd\n'
#     if ens == 'nvt' :        
#         ret += 'thermo_style    custom step ke pe etotal enthalpy temp press vol lx ly lz xy xz yz pxx pyy pzz pxy pxz pyz c_allmsd[*]\n'
#     elif 'npt' in ens :
#         ret += 'thermo_style    custom step ke pe etotal enthalpy temp press vol lx ly lz xy xz yz pxx pyy pzz pxy pxz pyz c_allmsd[*]\n'
#     else :	
#         raise RuntimeError('unknow ensemble %s\n' % ens)
#     if dump_ave_posi: 
#         ret += 'compute         ru all property/atom xu yu zu\n'
#         ret += 'fix             ap all ave/atom ${DUMP_FREQ} ${NREPEAT} ${NSTEPS} c_ru[1] c_ru[2] c_ru[3]\n'
#         ret += 'dump            fp all custom ${NSTEPS} dump.avgposi id type f_ap[1] f_ap[2] f_ap[3]\n'
#     ret += 'dump            1 all custom ${DUMP_FREQ} dump.equi id type x y z vx vy vz\n'
#     if ens == 'nvt' :
#         ret += 'fix             1 all nvt temp ${TEMP} ${TEMP} ${TAU_T}\n'
#     elif ens == 'npt-iso' or ens == 'npt':
#         ret += 'fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}\n'
#     elif ens == 'npt-xy' :
#         ret += 'fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} aniso ${PRES} ${PRES} ${TAU_P} couple xy\n'
#     elif ens == 'npt-aniso' :
#         ret += 'fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} aniso ${PRES} ${PRES} ${TAU_P}\n'
#     elif ens == 'npt-tri' :
#         ret += 'fix             1 all npt temp ${TEMP} ${TEMP} ${TAU_T} tri ${PRES} ${PRES} ${TAU_P}\n'
#     elif ens == 'nve' :
#         ret += 'fix             1 all nve\n'
#     else :
#         raise RuntimeError('unknow ensemble %s\n' % ens)        
#     ret += 'fix             mzero all momentum 10 linear 1 1 1\n'
#     ret += '# --------------------- INITIALIZE -----------------------\n'    
#     ret += 'velocity        all create ${TEMP} %d\n' % (np.random.randint(1, 2**16))
#     ret += 'velocity        all zero linear\n'
#     ret += '# --------------------- RUN ------------------------------\n'    
#     ret += 'run             ${NSTEPS}\n'
#     ret += 'write_data      out.lmp\n'
#     return ret

def npt_equi_conf(npt_dir) :
    thermo_file = os.path.join(npt_dir, 'log.lammps')
    dump_file = os.path.join(npt_dir, 'dump.equi')
    j_file = os.path.join(npt_dir, 'equi_settings.json')
    with open(j_file, 'r') as f:
        jdata = json.load(f)
    stat_skip = jdata['stat_skip']
    stat_bsize = jdata['stat_bsize']

    data = get_thermo(thermo_file)
    lx, lxe = block_avg(data[:, 8], skip = stat_skip, block_size = stat_bsize)
    ly, lye = block_avg(data[:, 9], skip = stat_skip, block_size = stat_bsize)
    lz, lze = block_avg(data[:,10], skip = stat_skip, block_size = stat_bsize)
    xy, xye = block_avg(data[:,11], skip = stat_skip, block_size = stat_bsize)
    xz, xze = block_avg(data[:,12], skip = stat_skip, block_size = stat_bsize)
    yz, yze = block_avg(data[:,13], skip = stat_skip, block_size = stat_bsize)
    # print('~~~', lx , ly, lz , xy, xz, yz)
    
    last_dump = get_last_dump(dump_file).split('\n')
    sys_data = system_data(last_dump)
    sys_data['cell'][0][0] = lx
    sys_data['cell'][1][1] = ly
    sys_data['cell'][2][2] = lz
    sys_data['cell'][1][0] = xy
    sys_data['cell'][2][0] = xz
    sys_data['cell'][2][1] = yz

    conf_lmp = from_system_data(sys_data)
    return conf_lmp

def extract(job_dir, output) :
    dump_file = os.path.join(job_dir, 'dump.avgposi')
    if os.path.isfile(dump_file) :
        print('# found dump.avgposi, use it')
    else :
        dump_file = os.path.join(job_dir, 'dump.equi')
        assert(os.path.isfile(dump_file))
        print('# found dump.equi, use it')        
    last_dump = get_last_dump(dump_file).split('\n')
    for idx in range(len(last_dump)) :
        ii = last_dump[idx]
        if 'ITEM: ATOMS' in ii :
            ii = ii.replace('f_ap[1]', 'x')
            ii = ii.replace('f_ap[2]', 'y')
            ii = ii.replace('f_ap[3]', 'z')
        last_dump[idx] = ii
    sys_data = system_data(last_dump)
    conf_lmp = from_system_data(sys_data)
    with open(output, 'w') as f:
        f.write(conf_lmp)
    # open(output, 'w').write(conf_lmp)

def make_task(iter_name, jdata, ens=None, temp=None, pres=None, if_dump_avg_posi=None, npt_dir=None):
    # jfile_path = os.path.abspath(jfile)
    # with open(jfile, 'r') as f:
    #     jdata = json.load(f)

    equi_args = [
        Argument("equi_conf", str),
        Argument("mass_map", list, alias=['model_mass_map']),
        Argument("model", str),
        Argument("nsteps", int),
        Argument("timestep", float, alias=['dt']),
        Argument("ens", str),
        Argument("temp", int),
        Argument("pres", int),
        Argument("tau_t", float),
        Argument("tau_p", float),
        Argument("thermo_freq", int, alias=['stat_freq']),
        Argument("dump_freq", int),
        Argument("stat_skip", int),
        Argument("stat_bsize", int),
        Argument("if_dump_avg_posi", bool, optional=True, default=False),
        Argument("is_water", bool, optional=True, default=False, alias=['if_water']),
        Argument("if_meam", bool, optional=True, default=False),
        Argument("meam_model", list, optional=True, default=False),
    ]

    equi_format = Argument("equi", dict, equi_args)

    equi_kwargs_settings = create_dict_not_empty_key(
        ens=ens,
        temp=temp,
        pres=pres,
        if_dump_avg_posi=if_dump_avg_posi,
        npt_dir=npt_dir
    )

    equi_pre_settings = jdata.copy()
    equi_pre_settings.update(equi_kwargs_settings)

    equi_settings = equi_format.normalize_value(equi_pre_settings)

    task_abs_dir = create_path(iter_name)

    if npt_dir is not None:
        npt_avg_conf_lmp = npt_equi_conf(npt_dir)
        with open(os.path.join(task_abs_dir, 'npt_avg.lmp'), 'w') as f:
            f.write(npt_avg_conf_lmp)
        equi_settings['equi_conf'] = 'npt_avg.lmp'
    else:
        equi_conf = equi_settings['equi_conf']
        relative_link_file(equi_conf, task_abs_dir)
        equi_settings['equi_conf'] = os.path.basename(equi_conf)

    model = equi_settings['model']
    if model:
        relative_link_file(model, task_abs_dir)
        equi_settings['model'] = os.path.basename(model)

    if_meam = equi_settings.get('if_meam', None)
    meam_model = equi_settings.get('meam_model', None)
    if if_meam:
        relative_link_file(meam_model['library'], task_abs_dir)
        relative_link_file(meam_model['potential'], task_abs_dir)


    with open(os.path.join(task_abs_dir, 'jdata.json'), 'w') as f:
        json.dump(jdata, f, indent=4)
    with open(os.path.join(task_abs_dir, 'equi_kwargs_setting.json'), 'w') as f:
        json.dump(equi_kwargs_settings, f, indent=4)
    with open(os.path.join(task_abs_dir, 'equi_settings.json'), 'w') as f:
        json.dump(equi_settings, f, indent=4)

    lmp_str = gen_equi_lammps_input(nsteps=equi_settings['nsteps'], 
        thermo_freq=equi_settings['thermo_freq'], 
        dump_freq=equi_settings['dump_freq'],
        mass_map=equi_settings['mass_map'],
        temp=equi_settings['temp'],
        tau_t=equi_settings['tau_t'],
        tau_p=equi_settings['tau_p'],
        equi_conf=equi_settings['equi_conf'],
        model=equi_settings['model'],
        timestep=equi_settings['timestep'],
        if_dump_avg_posi=equi_settings['if_dump_avg_posi'],
        ens=equi_settings['ens'],
        pres=equi_settings['pres'],
        if_meam=equi_settings['if_meam'], 
        meam_model=equi_settings['meam_model'])


    with open(os.path.join(task_abs_dir, 'in.lammps'), 'w') as fp :
        fp.write(lmp_str)
    return equi_settings


def water_bond(iter_name, skip = 1) :
    fdump = os.path.join(iter_name, 'dump.equi')
    with open(fdump, 'r') as f:
        lines = f.read().split('\n')
    # lines = open(fdump).read().split('\n')
    sections = []
    for ii in range(len(lines)) :
        if 'ITEM: TIMESTEP' in lines[ii] :
            sections.append(ii)
    sections.append(len(lines)-1) 
    all_rr = []
    all_tt = []
    for ii in range(skip, len(sections)-1) :
        sys_data = system_data(lines[sections[ii]:sections[ii+1]])
        atype = sys_data['atom_types']
        posis = sys_data['coordinates']
        cell  = sys_data['cell']
        bonds = compute_bonds(cell, atype, posis)
        if ii == skip : 
            bonds_0 = bonds 
        else :
            if bonds_0 != bonds :
                print('proton trans detected at frame %d' % ii)
        rr = []
        tt = []
        for ii in range(len(bonds)) :
            if atype[ii] == 1 :
                i_idx = ii
                j_idx = bonds[i_idx][0]
                k_idx = bonds[i_idx][1]
                drj = posi_diff(cell, posis[i_idx], posis[j_idx])
                drk = posi_diff(cell, posis[i_idx], posis[k_idx])
                ndrj = np.linalg.norm(drj)
                ndrk = np.linalg.norm(drk)
                rr.append(ndrj)
                rr.append(ndrk)
                tt.append(np.arccos(np.dot(drj,drk) / (ndrj*ndrk)))
        all_rr += rr
        all_tt += tt
    print('# statistics over %d frames %d angles' % (len(sections)-1, len(all_tt)))
    return (np.average(all_rr)), (np.average(all_tt))


def _compute_thermo (lmplog, natoms, stat_skip, stat_bsize) :
    # print(3939, natoms)
    data = get_thermo(lmplog)
    ea, ee = block_avg(data[:, 3], skip = stat_skip, block_size = stat_bsize)
    ha, he = block_avg(data[:, 4], skip = stat_skip, block_size = stat_bsize)
    ta, te = block_avg(data[:, 5], skip = stat_skip, block_size = stat_bsize)
    pa, pe = block_avg(data[:, 6], skip = stat_skip, block_size = stat_bsize)
    va, ve = block_avg(data[:, 7], skip = stat_skip, block_size = stat_bsize)
    lxx, lxxe = block_avg(data[:, 8], skip = stat_skip, block_size = stat_bsize)
    lyy, lyye = block_avg(data[:, 9], skip = stat_skip, block_size = stat_bsize)
    lzz, lzze = block_avg(data[:,10], skip = stat_skip, block_size = stat_bsize)
    lxy, lxye = block_avg(data[:,11], skip = stat_skip, block_size = stat_bsize)
    lxz, lxze = block_avg(data[:,12], skip = stat_skip, block_size = stat_bsize)
    lyz, lyze = block_avg(data[:,13], skip = stat_skip, block_size = stat_bsize)
    pxx, pxxe = block_avg(data[:,14], skip = stat_skip, block_size = stat_bsize)
    pyy, pyye = block_avg(data[:,15], skip = stat_skip, block_size = stat_bsize)
    pzz, pzze = block_avg(data[:,16], skip = stat_skip, block_size = stat_bsize)
    pxy, pxye = block_avg(data[:,17], skip = stat_skip, block_size = stat_bsize)
    pxz, pxze = block_avg(data[:,18], skip = stat_skip, block_size = stat_bsize)
    pyz, pyze = block_avg(data[:,19], skip = stat_skip, block_size = stat_bsize)
    thermo_info = {}
    thermo_info['p'] = pa
    thermo_info['p_err'] = pe
    thermo_info['v'] = va / natoms 
    thermo_info['v_err'] = ve / np.sqrt(natoms)
    thermo_info['e'] = ea / natoms 
    thermo_info['e_err'] = ee / np.sqrt(natoms)
    thermo_info['t'] = ta
    thermo_info['t_err'] = te
    thermo_info['h'] = ha / natoms 
    thermo_info['h_err'] = he / np.sqrt(natoms)
    unit_cvt = 1e5 * (1e-10**3) / pc.electron_volt
    thermo_info['pv'] = pa * va * unit_cvt / natoms
    thermo_info['pv_err'] = pe * va * unit_cvt / np.sqrt(natoms)
    thermo_info['lxx'] = lxx
    thermo_info['lxx_err'] = lxxe
    thermo_info['lyy'] = lyy
    thermo_info['lyy_err'] = lyye
    thermo_info['lzz'] = lzz
    thermo_info['lzz_err'] = lzze
    thermo_info['lxy'] = lxy
    thermo_info['lxy_err'] = lxye
    thermo_info['lxz'] = lxz
    thermo_info['lxz_err'] = lxze
    thermo_info['lyz'] = lyz
    thermo_info['lyz_err'] = lyze
    thermo_info['pxx'] = pxx
    thermo_info['pxx_err'] = pxxe
    thermo_info['pyy'] = pyy
    thermo_info['pyy_err'] = pyye
    thermo_info['pzz'] = pzz
    thermo_info['pzz_err'] = pzze
    thermo_info['pxy'] = pxy
    thermo_info['pxy_err'] = pxye
    thermo_info['pxz'] = pxz
    thermo_info['pxz_err'] = pxze
    thermo_info['pyz'] = pyz
    thermo_info['pyz_err'] = pyze
    return thermo_info

def _print_thermo_info(info, more_head = '') :
    ptr  = '# thermodynamics  %20s %20s  %s\n' % ('value', 'err', more_head)
    ptr += '# E        [eV]:  %20.8f %20.8f\n' % (info['e'], info['e_err'])
    ptr += '# H        [eV]:  %20.8f %20.8f\n' % (info['h'], info['h_err'])
    ptr += '# T         [K]:  %20.8f %20.8f\n' % (info['t'], info['t_err'])
    ptr += '# P       [bar]:  %20.8f %20.8f\n' % (info['p'], info['p_err'])
    ptr += '# V       [A^3]:  %20.8f %20.8f\n' % (info['v'], info['v_err'])
    ptr += '# PV       [eV]:  %20.8f %20.8f\n' % (info['pv'], info['pv_err'])
    ptr += '# Lxx       [A]:  %20.8f %20.8f\n' % (info['lxx'], info['lxx_err'])
    ptr += '# Lyy       [A]:  %20.8f %20.8f\n' % (info['lyy'], info['lyy_err'])
    ptr += '# Lzz       [A]:  %20.8f %20.8f\n' % (info['lzz'], info['lzz_err'])
    ptr += '# Lxy       [A]:  %20.8f %20.8f\n' % (info['lxy'], info['lxy_err'])
    ptr += '# Lxz       [A]:  %20.8f %20.8f\n' % (info['lxz'], info['lxz_err'])
    ptr += '# Lyz       [A]:  %20.8f %20.8f\n' % (info['lyz'], info['lyz_err'])
    ptr += '# Pxx     [bar]:  %20.8f %20.8f\n' % (info['pxx'], info['pxx_err'])
    ptr += '# Pyy     [bar]:  %20.8f %20.8f\n' % (info['pyy'], info['pyy_err'])
    ptr += '# Pzz     [bar]:  %20.8f %20.8f\n' % (info['pzz'], info['pzz_err'])
    ptr += '# Pxy     [bar]:  %20.8f %20.8f\n' % (info['pxy'], info['pxy_err'])
    ptr += '# Pxz     [bar]:  %20.8f %20.8f\n' % (info['pxz'], info['pxz_err'])
    ptr += '# Pyz     [bar]:  %20.8f %20.8f\n' % (info['pyz'], info['pyz_err'])
    # rho = (18 * 1e-3 / (info['v'] * pc.Avogadro * pc.angstrom**3))
    # rho_err = (info['v'] / (info['v'] - info['v_err'] ) - 1) * rho
    # ptr += '# water density [kg/m^3] : %10.5f (%10.5f)' % (rho, rho_err)
    print(ptr)
    return ptr

def post_task(iter_name, natoms = None, is_water = None) :
    # j_file = os.path.join(iter_name, 'in.json')
    j_file = os.path.join(iter_name, 'equi_settings.json')
    with open(j_file, 'r') as f:
        jdata = json.load(f)
    if natoms == None :
        equi_conf = get_task_file_abspath(iter_name, jdata['equi_conf'])
        natoms = get_natoms(equi_conf)
        if 'copies' in jdata :
            natoms *= np.prod(jdata['copies'])
    if is_water is None:
        is_water=jdata.get('is_water', False)
    else:
        pass
    if is_water is True:
        nmols = natoms // 3
    elif is_water is False:
        nmols = natoms
    else:
        raise RuntimeError('must specify key "is_water" of  bool value  in jdata')
    stat_skip = jdata['stat_skip']
    stat_bsize = jdata['stat_bsize']
    log_file = os.path.join(iter_name, 'log.lammps')
    info = _compute_thermo(log_file, nmols, stat_skip, stat_bsize)
    ptr = _print_thermo_info(info)

    info_dict = info.copy()
    out_lmp = os.path.abspath(os.path.join(iter_name, 'out.lmp'))
    info_dict['out_lmp'] = out_lmp
    info_dict['job_dir'] = iter_name

    with open(os.path.join(iter_name, 'result'), 'w') as f:
        f.write(ptr)
    # open(').write(ptr).close()
    with open(os.path.join(iter_name, 'result.json'), 'w') as f:
        json.dump(info_dict, f, indent=4)
        # f.write(json.dump(info))
    # open(, 'w').write(json.dumps(info)).close()
    return info_dict

def _main ():
    parser = argparse.ArgumentParser(
        description="Equilibrium simulation")
    subparsers = parser.add_subparsers(title='Valid subcommands', dest='command')

    parser_gen = subparsers.add_parser('gen', help='Generate a job')
    parser_gen.add_argument('PARAM', type=str ,
                            help='json parameter file')
    parser_gen.add_argument('-e','--ensemble', type=str,
                            help='the ensemble of the simulation')
    parser_gen.add_argument('-t','--temperature', type=float,
                            help='the temperature of the system')
    parser_gen.add_argument('-p','--pressure', type=float,
                            help='the pressure of the system')
    parser_gen.add_argument('-a','--avg-posi', action = 'store_true',
                            help='dump the average position of atoms')
    parser_gen.add_argument('-c','--conf-npt', type=str,
                            help='use conf computed from NPT simulation')
    parser_gen.add_argument('-o','--output', type=str, default = 'new_job',
                            help='the output folder for the job')
    # parser_gen.add_argument("-z", "--meam", help="whether use meam instead of dp", action="store_true")

    parser_comp = subparsers.add_parser('extract', help= 'Extract the conf')
    parser_comp.add_argument('JOB', type=str ,
                             help='folder of the job')
    parser_comp.add_argument('-o','--output', type=str, default = 'conf.lmp',
                             help='output conf file name')

    parser_stat = subparsers.add_parser('stat-bond', help= 'Statistic of the bonds')
    parser_stat.add_argument('JOB', type=str ,
                             help='folder of the job')
    parser_stat.add_argument('-s','--skip', type=int, default = 1,
                             help='skip this number of frames')

    parser_stat = subparsers.add_parser('compute', help= 'Compute thermodynamics')
    parser_stat.add_argument('JOB', type=str ,
                             help='folder of the job')

    args = parser.parse_args()

    
    if args.command is None :
        parser.print_help()
        exit
    if args.command == 'gen' :
        jdata = json.load(open(args.PARAM, 'r'))
        # jfile = os.path.abspath(args.PARAM)
        make_task(args.output, jdata, args.ensemble, args.temperature, args.pressure, args.avg_posi, args.conf_npt,)
    elif args.command == 'extract' :
        extract(args.JOB, args.output)
    elif args.command == 'stat-bond' :
        b, a = water_bond(args.JOB, args.skip)
        print(b, a/np.pi*180)
    elif args.command == 'compute' :
        post_task(args.JOB)


if __name__ == '__main__' :
    _main()
