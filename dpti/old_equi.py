#!/usr/bin/env python3

import os, sys, json, argparse, glob
import numpy as np
import scipy.constants as pc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from dpti.lib.utils import create_path
from dpti.lib.utils import block_avg
from dpti.lib.water import compute_bonds
from dpti.lib.water import posi_diff
from dpti.lib.utils import get_task_file_abspath
# import dpti

from dpti.lib.lammps import get_natoms, get_last_dump, get_thermo
from dpti.lib.lmp import from_system_data
from dpti.lib.dump import system_data
# from lib import dump 
# from .lib import lammps
# from .lib import dump
# from .lib import lmp

np.random.seed(0)

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
                       dump_freq = 1000, 
                       dump_ave_posi = False,
                       if_meam = False,
                       meam_model=None) :
    ret = ''
    ret += 'clear\n'
    ret += '# --------------------- VARIABLES-------------------------\n'
    ret += 'variable        NSTEPS          equal %d\n' % nsteps
    ret += 'variable        THERMO_FREQ     equal %d\n' % prt_freq
    ret += 'variable        DUMP_FREQ       equal %d\n' % dump_freq
    ret += 'variable        NREPEAT         equal ${NSTEPS}/${DUMP_FREQ}\n'
    ret += 'variable        TEMP            equal %f\n' % temp
    if pres is not None :
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
    if if_meam:
        ret += 'pair_style      meam\n'
        ret += f'pair_coeff      * * {meam_model[0]} {meam_model[2]} {meam_model[1]} {meam_model[2]}\n'
    else:
        ret += 'pair_style      deepmd %s\n' % model
        ret += 'pair_coeff\n'
    ret += '# --------------------- MD SETTINGS ----------------------\n'    
    ret += 'neighbor        1.0 bin\n'
    ret += 'timestep        %s\n' % dt
    ret += 'thermo          ${THERMO_FREQ}\n'
    ret += 'compute         allmsd all msd\n'
    if ens == 'nvt' :        
        ret += 'thermo_style    custom step ke pe etotal enthalpy temp press vol lx ly lz xy xz yz pxx pyy pzz pxy pxz pyz c_allmsd[*]\n'
    elif 'npt' in ens :
        ret += 'thermo_style    custom step ke pe etotal enthalpy temp press vol lx ly lz xy xz yz pxx pyy pzz pxy pxz pyz c_allmsd[*]\n'
    else :	
        raise RuntimeError('unknow ensemble %s\n' % ens)
    if dump_ave_posi: 
        ret += 'compute         ru all property/atom xu yu zu\n'
        ret += 'fix             ap all ave/atom ${DUMP_FREQ} ${NREPEAT} ${NSTEPS} c_ru[1] c_ru[2] c_ru[3]\n'
        ret += 'dump            fp all custom ${NSTEPS} dump.avgposi id type f_ap[1] f_ap[2] f_ap[3]\n'
    ret += 'dump            1 all custom ${DUMP_FREQ} dump.equi id type x y z vx vy vz\n'
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

def npt_equi_conf(npt_name) :
    thermo_file = os.path.join(npt_name, 'log.lammps')
    dump_file = os.path.join(npt_name, 'dump.equi')
    j_file = os.path.join(npt_name, 'in.json')
    jdata = json.load(open(j_file))
    stat_skip = jdata['stat_skip']
    stat_bsize = jdata['stat_bsize']

    data = get_thermo(thermo_file)
    lx, lxe = block_avg(data[:, 8], skip = stat_skip, block_size = stat_bsize)
    ly, lye = block_avg(data[:, 9], skip = stat_skip, block_size = stat_bsize)
    lz, lze = block_avg(data[:,10], skip = stat_skip, block_size = stat_bsize)
    xy, xye = block_avg(data[:,11], skip = stat_skip, block_size = stat_bsize)
    xz, xze = block_avg(data[:,12], skip = stat_skip, block_size = stat_bsize)
    yz, yze = block_avg(data[:,13], skip = stat_skip, block_size = stat_bsize)
    print('~~~', lx , ly, lz , xy, xz, yz)
    
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
    open(output, 'w').write(conf_lmp)

def make_task(iter_name, jdata, ens=None, temp=None, pres=None, avg_posi=None, npt_conf=None, if_meam=None):
    equi_conf = jdata['equi_conf']
    equi_conf = os.path.abspath(equi_conf)
    if npt_conf is not None :
        npt_conf = os.path.abspath(npt_conf)
    model = jdata['model']
    meam_model = jdata.get('meam_model', None)
    model = os.path.abspath(model)
    model_mass_map = jdata['model_mass_map']
    nsteps = jdata['nsteps']
    dt = jdata['dt']
    stat_freq = jdata['stat_freq']
    dump_freq = jdata['dump_freq']
    tau_t = jdata['tau_t']
    tau_p = jdata['tau_p']

    if ens is None :
        ens = jdata['ens']
    elif 'ens' in jdata :
        print('ens = %s overrides the ens in json data' % ens)
    jdata['ens'] = ens
    if temp is None :
        temp = jdata['temp']
    elif 'temp' in jdata :
        print('T = %f overrides the temp in json data' % temp)
    jdata['temp'] = temp
    if 'npt' in ens :
        if pres == None :
            pres = jdata['pres']
        elif 'pres' in jdata :
            print('P = %f overrides the pres in json data' % pres)
        jdata['pres'] = pres    
    if if_meam is None:
        if_meam = jdata.get('if_meam', False)

    create_path(iter_name)
    cwd = os.getcwd()
    os.chdir(iter_name)
    with open('in.json', 'w') as fp:
        json.dump(jdata, fp, indent=4)
    if npt_conf is None :
        os.symlink(os.path.realpath(equi_conf), 'conf.lmp')
    else :        
        open('conf.lmp', 'w').write(npt_equi_conf(npt_conf))
    os.symlink(os.path.realpath(model), 'graph.pb')
    lmp_str \
            = _gen_lammps_input('conf.lmp',
                            model_mass_map, 
                            'graph.pb',
                            nsteps, 
                            dt, 
                            ens, 
                            temp,
                            pres, 
                            tau_t = tau_t,
                            tau_p = tau_p,
                            prt_freq = stat_freq, 
                            dump_freq = dump_freq, 
                            dump_ave_posi = avg_posi,
                            if_meam = if_meam,
                            meam_model = meam_model)
    with open('in.lammps', 'w') as fp :
        fp.write(lmp_str)
    os.chdir(cwd)

def water_bond(iter_name, skip = 1) :
    fdump = os.path.join(iter_name, 'dump.equi')
    lines = open(fdump).read().split('\n')
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
    rho = (18 * 1e-3 / (info['v'] * pc.Avogadro * pc.angstrom**3))
    rho_err = (info['v'] / (info['v'] - info['v_err'] ) - 1) * rho
    ptr += '# water density [kg/m^3] : %10.5f (%10.5f)' % (rho, rho_err)
    print(ptr)
    return ptr

def post_task(iter_name, natoms = None, is_water = False) :
    j_file = os.path.join(iter_name, 'in.json')
    jdata = json.load(open(j_file))
    if natoms == None :
        equi_conf = get_task_file_abspath(iter_name, jdata['equi_conf'])
        natoms = get_natoms(equi_conf)
        if 'copies' in jdata :
            natoms *= np.prod(jdata['copies'])
    is_water=jdata.get('is_water', True)
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
    open(os.path.join(iter_name, 'result'), 'w').write(ptr)
    open(os.path.join(iter_name, 'result.json'), 'w').write(json.dumps(info))
    return info

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
    parser_gen.add_argument("-z", "--meam", help="whether use meam instead of dp", action="store_true")

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
        make_task(args.output, jdata, args.ensemble, args.temperature, args.pressure, args.avg_posi, args.conf_npt, if_meam=args.meam)
    elif args.command == 'extract' :
        extract(args.JOB, args.output)
    elif args.command == 'stat-bond' :
        b, a = water_bond(args.JOB, args.skip)
        print(b, a/np.pi*180)
    elif args.command == 'compute' :
        post_task(args.JOB)


if __name__ == '__main__' :
    _main()
