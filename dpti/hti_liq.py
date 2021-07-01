#!/usr/bin/env python3

import os, sys, json, argparse, glob, shutil
import numpy as np
import scipy.constants as pc

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from dpti import einstein
import dpti.lib.lmp as lmp
from dpti.lib.utils import create_path, relative_link_file
from dpti.lib.utils import copy_file_list
from dpti.lib.utils import block_avg
from dpti.lib.utils import integrate
from dpti.lib.utils import integrate_sys_err
from dpti.lib.utils import parse_seq
from dpti.lib.utils import get_first_matched_key_from_dict
from dpti.lib.lammps import get_thermo
from dpti.lib.lammps import get_natoms

def make_iter_name (iter_index) :
    return "task_hti." + ('%04d' % iter_index)

def _ff_soft_on(lamb, 
                sparam) :
    nn = sparam['n']
    alpha_lj = sparam['alpha_lj']
    rcut = sparam['rcut']
    epsilon = sparam['epsilon']
    # sigma = sparam['sigma']
    activation = sparam['activation']
    ret = ''
    ret += 'variable        EPSILON equal %f\n' % epsilon
    ret += 'pair_style      lj/cut/soft %f %f %f\n' % (nn, alpha_lj, rcut)

    element_num=sparam.get('element_num', 1)
    sigma_key_index = filter(lambda t:t[0] <= t[1], ((i,j) for i in range(element_num) for j in range(element_num)))
    for (i, j) in sigma_key_index:
        ret += 'pair_coeff      %s %s ${EPSILON} %f %f\n' % (i+1, j+1, sparam['sigma_'+str(i)+'_'+str(j)], activation)

    # ret += 'pair_coeff      * * ${EPSILON} %f %f\n' % (sigma, activation)
    ret += 'fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_LAMBDA scale yes\n'
    ret += 'compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_EPSILON\n'
    return ret

def _ff_deep_on(lamb, 
                sparam, 
                model,
                if_meam=False,
                meam_model=None):
    nn = sparam['n']
    alpha_lj = sparam['alpha_lj']
    rcut = sparam['rcut']
    epsilon = sparam['epsilon']
    # sigma = sparam['sigma']
    activation = sparam['activation']
    ret = ''
    ret += 'variable        EPSILON equal %f\n' % epsilon
    ret += 'variable        ONE equal 1\n'
    if if_meam:
        ret += 'pair_style      hybrid/overlay meam lj/cut/soft %f %f %f\n' % (nn, alpha_lj, rcut)
        ret += f'pair_coeff      * * meam {meam_model["library"]} {meam_model["element"]} {meam_model["potential"]} {meam_model["element"]}\n'
        # ret += f'pair_coeff      * * meam {meam_model[0]} {meam_model[2]} {meam_model[1]} {meam_model[2]}\n'
    else:
        ret += 'pair_style      hybrid/overlay deepmd %s lj/cut/soft %f %f %f\n' % (model, nn, alpha_lj, rcut)
        ret += 'pair_coeff      * * deepmd\n'

    element_num=sparam.get('element_num', 1)
    sigma_key_index = filter(lambda t:t[0] <= t[1], ((i,j) for i in range(element_num) for j in range(element_num)))
    for (i, j) in sigma_key_index:
        ret += 'pair_coeff      %s %s lj/cut/soft ${EPSILON} %f %f\n' % (i+1, j+1, sparam['sigma_'+str(i)+'_'+str(j)], activation)

    # ret += 'pair_coeff      * * lj/cut/soft ${EPSILON} %f %f\n' % (sigma, activation)
    if if_meam:
        ret += 'fix             tot_pot all adapt/fep 0 pair meam scale * * v_LAMBDA\n'
        ret += 'compute         e_diff all fep ${TEMP} pair meam scale * * v_ONE\n'
    else:
        ret += 'fix             tot_pot all adapt/fep 0 pair deepmd scale * * v_LAMBDA\n'
        ret += 'compute         e_diff all fep ${TEMP} pair deepmd scale * * v_ONE\n'
    return ret

def _ff_soft_off(lamb, 
                 sparam, 
                 model,
                 if_meam=False,
                 meam_model=None) :
    nn = sparam['n']
    alpha_lj = sparam['alpha_lj']
    rcut = sparam['rcut']
    epsilon = sparam['epsilon']
    # sigma = sparam['sigma']
    activation = sparam['activation']
    ret = ''
    ret += 'variable        INV_LAMBDA equal 1-${LAMBDA}\n'
    ret += 'variable        EPSILON equal %f\n' % epsilon
    ret += 'variable        INV_EPSILON equal -${EPSILON}\n'
    if if_meam:
        ret += 'pair_style      hybrid/overlay meam lj/cut/soft %f %f %f\n' % (nn, alpha_lj, rcut)
        ret += f'pair_coeff      * * meam {meam_model["library"]} {meam_model["element"]} {meam_model["potential"]} {meam_model["element"]}\n'
        # ret += f'pair_coeff      * * meam {meam_model[0]} {meam_model[2]} {meam_model[1} {meam_model[2]} \n'
    else:
        ret += 'pair_style      hybrid/overlay deepmd %s lj/cut/soft %f %f %f\n' % (model, nn, alpha_lj, rcut)
        ret += 'pair_coeff      * * deepmd\n'

    element_num=sparam.get('element_num', 1)
    sigma_key_index = filter(lambda t:t[0] <= t[1], ((i,j) for i in range(element_num) for j in range(element_num)))
    for (i, j) in sigma_key_index:
        ret += 'pair_coeff      %s %s lj/cut/soft ${EPSILON} %f %f\n' % (i+1, j+1, sparam['sigma_'+str(i)+'_'+str(j)], activation)

    # ret += 'pair_coeff      * * lj/cut/soft ${EPSILON} %f %f\n' % (sigma, activation)
    ret += 'fix             tot_pot all adapt/fep 0 pair lj/cut/soft epsilon * * v_INV_LAMBDA scale yes\n'
    ret += 'compute         e_diff all fep ${TEMP} pair lj/cut/soft epsilon * * v_INV_EPSILON\n'
    return ret

def _gen_lammps_input_ideal (step,
                             conf_file, 
                             mass_map,
                             lamb,
                             soft_param,
                             model,
                             nsteps,
                             timestep,
                             ens,
                             temp,
                             pres = 1.0, 
                             tau_t = 0.1,
                             tau_p = 0.5,
                             thermo_freq = 100, 
                             copies = None,
                             norm_style = 'first',
                             if_meam = False,
                             meam_model = None) :
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
    if step == 'soft_on' :
        ret += _ff_soft_on(lamb, soft_param)
    elif step == 'deep_on' :
        ret += _ff_deep_on(lamb, soft_param, model, if_meam=if_meam, meam_model=meam_model)
    elif step == 'soft_off' :
        ret += _ff_soft_off(lamb, soft_param, model, if_meam=if_meam, meam_model=meam_model)
    else :
        raise RuntimeError('unknown step')
    ret += '# --------------------- MD SETTINGS ----------------------\n'    
    ret += 'neighbor        1.0 bin\n'
    ret += 'timestep        %s\n' % timestep
    ret += 'compute         allmsd all msd\n'
    ret += 'thermo          ${THERMO_FREQ}\n'
    ret += 'thermo_style    custom step ke pe etotal enthalpy temp press vol c_e_diff[1] c_allmsd[*]\n'
    ret += 'thermo_modify   format 9 %.16e\n'
    ret += '# dump            1 all custom ${DUMP_FREQ} dump.hti id type x y z vx vy vz\n'
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


def _make_tasks(iter_name, jdata, step, if_meam=False, meam_model=None) :
    if step == 'soft_on' :
        all_lambda = parse_seq(jdata['lambda_soft_on'])
    elif step == 'deep_on' :
        all_lambda = parse_seq(jdata['lambda_deep_on'])
    elif step == 'soft_off' :
        all_lambda = parse_seq(jdata['lambda_soft_off'])
    else :
        raise RuntimeError('unknow step')
    equi_conf = jdata['equi_conf']
    equi_conf = os.path.abspath(equi_conf)
    # mass_map = jdata['mass_map']
    mass_map = get_first_matched_key_from_dict(jdata, ['mass_map', 'model_mass_map'])
    model = jdata.get('model', None)
    if model:
        model = os.path.abspath(model)
    soft_param = jdata['soft_param']
    nsteps = jdata['nsteps']
    # timestep = jdata['timestep']
    timestep = get_first_matched_key_from_dict(jdata, ['timestep', 'dt'])
    # thermo_freq = jdata['thermo_freq']
    thermo_freq = get_first_matched_key_from_dict(jdata, ['thermo_freq', 'stat_freq'])
    copies = None
    if 'copies' in jdata :
        copies = jdata['copies']
    temp = jdata['temp']
    
    sparam = jdata.get('soft_param', {})
    if sparam:
        if 'sigma_oo' in sparam:
            sparam['sigma_0_0'] = sparam['sigma_oo']
            sparam['sigma_0_1'] = sparam['sigma_oh']
            sparam['sigma_1_1'] = sparam['sigma_hh']

        # element_num=sparam.get('element_num', 1)
        element_num = len(mass_map)
        sigma_key_index = filter(lambda t:t[0] <= t[1], ((i,j) for i in range(element_num) for j in range(element_num)))
        sigma_key_name_list = ['sigma_'+str(t[0])+'_'+str(t[1]) for t in sigma_key_index ]
        for sigma_key_name in sigma_key_name_list:
            assert sparam.get(sigma_key_name, None), 'there must be key-value for {sigma_key_name} in soft_param'.format(sigma_key_name=sigma_key_name)

    job_abs_dir = create_path(iter_name)

    if meam_model:
        relative_link_file(os.path.abspath(meam_model['library']), job_abs_dir)
        relative_link_file(os.path.abspath(meam_model['potential']), job_abs_dir)
        # os.symlink(os.path.join('..', 'conf.lmp'), 'conf.lmp')
        # os.symlink(os.path.join('..', 'graph.pb'), 'graph.pb')

    cwd = os.getcwd()
    os.chdir(iter_name)
    os.symlink(os.path.join('..', 'in.json'), 'in.json')
    os.symlink(os.path.join('..', 'conf.lmp'), 'conf.lmp')
    os.symlink(os.path.join('..', 'graph.pb'), 'graph.pb')

    os.chdir(cwd)
    # print(9898, meam_model)
    for idx,ii in enumerate(all_lambda) :
        work_path = os.path.join(iter_name, 'task.%06d' % idx)
        create_path(work_path)
        os.chdir(work_path)
        os.symlink(os.path.join('..', 'conf.lmp'), 'conf.lmp')
        os.symlink(os.path.join('..', 'graph.pb'), 'graph.pb')
        if meam_model:
            meam_library_basename = os.path.basename(meam_model['library'])
            meam_potential_basename = os.path.basename(meam_model['potential'])
            os.symlink(os.path.join('..', meam_library_basename), meam_library_basename)
            os.symlink(os.path.join('..', meam_potential_basename), meam_potential_basename)
        lmp_str \
            = _gen_lammps_input_ideal(step, 
                                      'conf.lmp',
                                      mass_map, 
                                      ii, 
                                      soft_param,
                                      'graph.pb',
                                      nsteps, 
                                      timestep,
                                      'nvt',
                                      temp,
                                      thermo_freq = thermo_freq, 
                                      copies = copies,
                                      if_meam = if_meam,
                                      meam_model=meam_model)
        with open('in.lammps', 'w') as fp :
            fp.write(lmp_str)
        with open('lambda.out', 'w') as fp :
            fp.write(str(ii))
        os.chdir(cwd)


def make_tasks(iter_name, jdata, if_meam=None) :
    if not if_meam :
        if_meam = jdata['if_meam']
    equi_conf = os.path.abspath(jdata['equi_conf'])
    if jdata.get('model', None):
        model = os.path.abspath(jdata['model'])
    else:
        model = None
    meam_model = jdata.get('meam_model', None)

    create_path(iter_name)
    copied_conf = os.path.join(os.path.abspath(iter_name), 'conf.lmp')
    shutil.copyfile(equi_conf, copied_conf)
    # jdata['equi_conf'] = copied_conf
    if model:
        copied_model = os.path.join(os.path.abspath(iter_name), 'graph.pb')
        shutil.copyfile(model, copied_model)
    # jdata['model'] = copied_model

    cwd = os.getcwd()
    os.chdir(iter_name)    
    with open('in.json', 'w') as fp:
        json.dump(jdata, fp, indent=4)
    os.chdir(cwd)
    subtask_name = os.path.join(iter_name, '00.soft_on')
    _make_tasks(subtask_name, jdata, 'soft_on', if_meam=if_meam, meam_model=meam_model)
    subtask_name = os.path.join(iter_name, '01.deep_on')
    _make_tasks(subtask_name, jdata, 'deep_on', if_meam=if_meam, meam_model=meam_model)
    subtask_name = os.path.join(iter_name, '02.soft_off')
    _make_tasks(subtask_name, jdata, 'soft_off', if_meam=if_meam, meam_model=meam_model)


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


def _post_tasks(iter_name, step, natoms) :
    jdata = json.load(open(os.path.join(iter_name, 'in.json')))
    stat_skip = jdata['stat_skip']
    stat_bsize = jdata['stat_bsize']
    all_tasks = glob.glob(os.path.join(iter_name, 'task.[0-9]*'))
    all_tasks.sort()
    ntasks = len(all_tasks)
    
    all_lambda = []
    all_dp_a = []
    all_dp_e = []
    all_msd_xyz = []

    for ii in all_tasks :
        log_name = os.path.join(ii, 'log.lammps')
        data = get_thermo(log_name)
        np.savetxt(os.path.join(ii, 'data'), data, fmt = '%.6e')
        dp_a, dp_e = block_avg(data[:, 8], skip = stat_skip, block_size = stat_bsize)
        msd_xyz = data[-1, 12]
        dp_a /= natoms
        dp_e /= np.sqrt(natoms)
        lmda_name = os.path.join(ii, 'lambda.out')
        ll = float(open(lmda_name).read())
        all_lambda.append(ll)
        all_dp_a.append(dp_a)
        all_dp_e.append(dp_e)
        all_msd_xyz.append(msd_xyz)

    all_lambda = np.array(all_lambda)
    all_dp_a = np.array(all_dp_a)
    all_dp_e = np.array(all_dp_e)
    all_msd_xyz = np.array(all_msd_xyz)
    de = all_dp_a
    all_err = all_dp_e

    all_print = []
    # all_print.append(np.arange(len(all_lambda)))
    all_print.append(all_lambda)
    all_print.append(de)
    all_print.append(all_err)
    all_print.append(all_msd_xyz)
    all_print = np.array(all_print)
    np.savetxt(os.path.join(iter_name, 'hti.out'), 
               all_print.T, 
               fmt = '%.8e', 
               header = 'lmbda dU dU_err msd_xyz')

    diff_e, err = integrate(all_lambda, de, all_err)
    sys_err = integrate_sys_err(all_lambda, de)

    thermo_info = _compute_thermo(os.path.join(all_tasks[-1], 'log.lammps'), 
                                  natoms,
                                  stat_skip, stat_bsize)

    return diff_e, [err, sys_err], thermo_info


def post_tasks(iter_name, natoms) :
    fe = einstein.ideal_gas_fe(iter_name)
    subtask_name = os.path.join(iter_name, '00.soft_on')
    e0, err0, tinfo0 = _post_tasks(subtask_name, 'soft_on', natoms)
    subtask_name = os.path.join(iter_name, '01.deep_on')
    e1, err1, tinfo1 = _post_tasks(subtask_name, 'deep_on', natoms)
    subtask_name = os.path.join(iter_name, '02.soft_off')
    e2, err2, tinfo2 = _post_tasks(subtask_name, 'soft_off', natoms)
    fe = fe + e0 + e1 + e2
    print(f'# HTI three-step error [stt_err, sys_err] {err0} {err1} {err2}')
    err = np.sqrt(np.square(err0[0]) + np.square(err1[0]) + np.square(err2[0]))
    sys_err = ((err0[1]) + (err1[1]) + (err2[1]))
    return fe, [err,sys_err], tinfo2


def _print_thermo_info(info) :
    ptr = '# thermodynamics (normalized by nmols)\n'
    ptr += '# E (err)  [eV]:  %20.8f %20.8f\n' % (info['e'], info['e_err'])
    ptr += '# H (err)  [eV]:  %20.8f %20.8f\n' % (info['h'], info['h_err'])
    ptr += '# T (err)   [K]:  %20.8f %20.8f\n' % (info['t'], info['t_err'])
    ptr += '# P (err) [bar]:  %20.8f %20.8f\n' % (info['p'], info['p_err'])
    ptr += '# V (err) [A^3]:  %20.8f %20.8f\n' % (info['v'], info['v_err'])
    ptr += '# PV(err)  [eV]:  %20.8f %20.8f' % (info['pv'], info['pv_err'])
    print(ptr)

def compute_task(job, free_energy_type='helmholtz', scheme='simpson', manual_pv=None, manual_pv_err=None):
    jdata = json.load(open(os.path.join(job, 'in.json'), 'r'))
    fp_conf = open(os.path.join(job, 'conf.lmp'))
    sys_data = lmp.to_system_data(fp_conf.read().split('\n'))
    natoms = sum(sys_data['atom_numbs'])
    jdata = json.load(open(os.path.join(job, 'in.json'), 'r'))
    if 'copies' in jdata :
        natoms *= np.prod(jdata['copies'])
    fe, fe_err, thermo_info = post_tasks(job, natoms)
    _print_thermo_info(thermo_info)

    info = thermo_info.copy()

    pv = None
    pv_err = None
    
    print ('# numb atoms: %d' % natoms)
    print_format = '%20.12f  %10.3e  %10.3e'
    if free_energy_type == 'helmholtz' :
        e1 = fe # e0 + de
        e1_err = fe_err[0]
        print('# Helmholtz free ener per atom (err) [eV]:')
        print(print_format % (fe, fe_err[0], fe_err[1]))
    if free_energy_type == 'gibbs' :
        if manual_pv is None:
            pv = thermo_info['pv']
        else: 
            pv = manual_pv
        if manual_pv_err is None:
            pv_err = thermo_info['pv_err']
        else:
            pv_err = manual_pv_err
        e1 = fe + pv
        e1_err = np.sqrt(fe_err[0]**2 + pv_err**2)
        print('# Gibbs free ener per mol (err) [eV]:')
        print(print_format % (e1, e1_err, fe_err[1]))
    else:
        raise RuntimeError('known free energy type')

    info['free_energy_type'] = free_energy_type
    info['pv'] = pv
    info['pv_err'] = pv_err
    # info['de'] = de
    # info['de_err'] = de_err
    info['e1'] = e1
    info['e1_err'] = e1_err
    open(os.path.join(job, 'result.json'), 'w').write(json.dumps(info))
    return info

def _main ():
    parser = argparse.ArgumentParser(
        description="Compute liquid free energy by Hamiltonian TI")
    subparsers = parser.add_subparsers(title='Valid subcommands', dest='command')

    parser_gen = subparsers.add_parser('gen', help='Generate a job')
    parser_gen.add_argument('PARAM', type=str ,
                            help='json parameter file')
    parser_gen.add_argument('-o','--output', type=str, default = 'new_job',
                            help='the output folder for the job')
    parser_gen.add_argument("-z", "--meam", help="whether use meam instead of dp", action="store_true")

    parser_comp = subparsers.add_parser('compute', help= 'Compute the result of a job')
    parser_comp.add_argument('JOB', type=str ,
                             help='folder of the job')
    parser_comp.add_argument('-t','--type', type=str, default = 'helmholtz', 
                             choices=['helmholtz', 'gibbs'], 
                             help='the type of free energy')
    parser_comp.add_argument('-g', '--pv', type=float, default = None,
                             help='press*vol value override to calculate Gibbs free energy')
    parser_comp.add_argument('-G', '--pv-err', type=float, default = None,
                             help='press*vol error')
    args = parser.parse_args()

    if args.command is None :
        parser.print_help()
        exit
    if args.command == 'gen' :
        output = args.output
        jdata = json.load(open(args.PARAM, 'r'))
        make_tasks(output, jdata, if_meam=args.meam)
    elif args.command == 'compute' :
        compute_task(job=args.JOB, free_energy_type=args.type, manual_pv=args.pv, manual_pv_err=args.pv_err)

     #    fp_conf = open(os.path.join(args.JOB, 'conf.lmp'))
     #    sys_data = lmp.to_system_data(fp_conf.read().split('\n'))
     #    natoms = sum(sys_data['atom_numbs'])
     #    jdata = json.load(open(os.path.join(args.JOB, 'in.json'), 'r'))
     #    if 'copies' in jdata :
     #        natoms *= np.prod(jdata['copies'])
     #    fe, fe_err, thermo_info = post_tasks(args.JOB, natoms)
     #    _print_thermo_info(thermo_info)
     #    print ('# numb atoms: %d' % natoms)
     #    print_format = '%20.12f  %10.3e  %10.3e'
     #    if args.type == 'helmholtz' :
     #        print('# Helmholtz free ener per atom (err) [eV]:')
     #        print(print_format % (fe, fe_err[0], fe_err[1]))
     #    if args.type == 'gibbs' :
     #        if args.pv is None:
     #            pv = thermo_info['pv']
     #        else: 
     #            pv = args.pv
     #        if args.pv_err is None:
     #            pv_err = thermo_info['pv_err']
     #        else:
     #            pv_err = args.pv_err
     #        e1 = fe + pv
     #        e1_err = np.sqrt(fe_err[0]**2 + pv_err**2)
     #        print('# Gibbs free ener per mol (err) [eV]:')
     #        print(print_format % (e1, e1_err, fe_err[1]))

    
if __name__ == '__main__' :
    _main()
