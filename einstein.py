#!/usr/bin/env python3

import os, sys, json, argparse, glob
import numpy as np
import scipy.constants as pc
import lib.lmp

def compute_lambda(temp, mass) :
    ret = 2. * np.pi * mass * (1e-3 / pc.Avogadro) * pc.Boltzmann * temp / (pc.Planck * pc.Planck)
    return 1./np.sqrt(ret)    

def compute_spring(temp, spring_k) :
    ret = (0.5 * spring_k * pc.electron_volt / (pc.angstrom * pc.angstrom)) / (pc.Boltzmann * temp * np.pi) 
    return np.sqrt(ret)

def ideal_gas_fe(job) :
    jdata = json.load(open(os.path.join(job, 'in.json'), 'r'))    
    equi_conf = jdata['equi_conf']
    cwd = os.getcwd()
    os.chdir(job)
    assert(os.path.isfile(equi_conf))
    equi_conf = os.path.abspath(equi_conf)
    os.chdir(cwd)
    temp = jdata['temp']
    mass_map = jdata['model_mass_map']
    if 'copies' in jdata :
        ncopies = np.prod(jdata['copies'])
    else :
        ncopies = 1

    sys_data = lib.lmp.to_system_data(open(equi_conf).read().split('\n'))
    vol = np.linalg.det(sys_data['cell'])
    natoms = [ii * ncopies for ii in sys_data['atom_numbs']]

    Lambda_k = [compute_lambda(temp, ii) for ii in mass_map]    
    fe = 0
    for idx,ii in enumerate(natoms) :
        # kinetic contrib
        # print('```', idx, ii)
        if ii > 0:
            rho = ii / (vol * (pc.angstrom**3))
            fe += ii * np.log(rho * (Lambda_k[idx] ** 3)) 
            fe -= ii
            fe += 0.5 * np.log(2. * np.pi * ii)
    fe *= pc.Boltzmann * temp / pc.electron_volt
    fe /= np.sum(natoms)
    return fe

def free_energy (job) :
    jdata = json.load(open(os.path.join(job, 'in.json'), 'r'))    
    equi_conf = jdata['equi_conf']
    cwd = os.getcwd()
    os.chdir(job)
    assert(os.path.isfile(equi_conf))
    equi_conf = os.path.abspath(equi_conf)
    os.chdir(cwd)
    temp = jdata['temp']
    mass_map = jdata['model_mass_map']
    spring_k = jdata['spring_k']
    if type(spring_k) is not list:
        spring_k_1 = []
        for ii in mass_map :
            spring_k_1.append(spring_k * ii)
        spring_k = spring_k_1
    assert(len(mass_map) == len(spring_k))
    if 'copies' in jdata :
        ncopies = np.prod(jdata['copies'])
    else :
        ncopies = 1

    sys_data = lib.lmp.to_system_data(open(equi_conf).read().split('\n'))
    vol = np.linalg.det(sys_data['cell'])
    natoms = [ii * ncopies for ii in sys_data['atom_numbs']]
    
    Lambda_k = [compute_lambda(temp, ii) for ii in mass_map]
    Lambda_s = [compute_spring(temp, ii) for ii in spring_k]
    # print(np.log(Lambda_k), np.log(Lambda_s))
    
    with open(equi_conf) as fp:
        lines = list(fp)
    for idx,ii in enumerate(lines) :
        if 'Atoms' in ii :
            break
    first_type = int(lines[idx+2].split()[1]) - 1
    print('# fixed atom of type %d ' % first_type)

    fe = 0
    fact = pc.Boltzmann * temp / pc.electron_volt / np.sum(natoms)
    for idx,ii in enumerate(natoms) :
        # kinetic contrib
        # print(idx)
        fe += 3 * ii * np.log(Lambda_k[idx])
        # print(3 * ii * np.log(Lambda_k[idx]) * fact)
        if (idx == first_type) :            
            fe += 3 * (ii-1) * np.log(Lambda_s[idx])
            fe += np.log(ii / (vol * (pc.angstrom**3)))
            # print(3.0 * (ii-1) * np.log(Lambda_s[idx]) * fact)
            # print(np.log(ii / (vol * (pc.angstrom**3))) * fact)
        else :
            fe += 3 * ii * np.log(Lambda_s[idx])
            # print(3.0 * ii * np.log(Lambda_s[idx]) * fact)
    fe *= pc.Boltzmann * temp / pc.electron_volt
    fe /= np.sum(natoms)
    # print(fe)
    return fe


def frenkel(job) :
    jdata = json.load(open(os.path.join(job, 'in.json'), 'r'))    
    equi_conf = jdata['equi_conf']
    cwd = os.getcwd()
    os.chdir(job)
    assert(os.path.isfile(equi_conf))
    equi_conf = os.path.abspath(equi_conf)
    os.chdir(cwd)
    temp = jdata['temp']
    mass_map = jdata['model_mass_map']
    s_spring_k = jdata['spring_k']
    spring_k = jdata['spring_k']
    assert(type(spring_k) is not list)
    if type(spring_k) is not list:
        spring_k_1 = []
        for ii in mass_map :
            spring_k_1.append(spring_k * ii)
        spring_k = spring_k_1
    if 'copies' in jdata :
        ncopies = np.prod(jdata['copies'])
    else :
        ncopies = 1    

    sys_data = lib.lmp.to_system_data(open(equi_conf).read().split('\n'))
    vol = np.linalg.det(sys_data['cell'])
    natoms = [ii * ncopies for ii in sys_data['atom_numbs']]

    Lambda_k = [compute_lambda(temp, ii) for ii in mass_map]
    Lambda_s = [compute_spring(temp, ii) for ii in spring_k]
    s_Lambda_s = compute_spring(temp, s_spring_k)

    fe = 0
    sum_m = 0
    fact = pc.Boltzmann * temp / pc.electron_volt / np.sum(natoms)
    for idx,ii in enumerate(natoms) :
        fe += 3.0 * ii * np.log(Lambda_k[idx])
        fe += 3.0 * ii * np.log(Lambda_s[idx])
        # print(idx)
        # print(3.0 * ii * np.log(Lambda_k[idx]) * fact)
        # print(3.0 * ii * np.log(Lambda_s[idx]) * fact)
        sum_m += mass_map[idx] * ii
    fe -= 3.0 * np.log(s_Lambda_s)
    fe -= 1.5 * np.log(sum_m)
    # fe += 2.0 * np.log(np.sum(natoms)/3.0)
    print('# FS corr (does not apply)', 2.0 * np.log(np.sum(natoms)/3.0) *pc.Boltzmann * temp / pc.electron_volt / np.sum(natoms) * 3.0)
    # print((3.0 * np.log(s_Lambda_s) + 1.5 * np.log(sum_m)) * fact)
    fe += np.log(np.sum(natoms) / (vol * (pc.angstrom**3)))    
    # print(np.log(np.sum(natoms) / (vol * (pc.angstrom**3))) * fact, np.log(np.sum(natoms) / 3.0 / (vol * (pc.angstrom**3))) * fact)

    fe *= pc.Boltzmann * temp / pc.electron_volt
    fe /= np.sum(natoms)
    return fe
        
    
def _main() :
    parser = argparse.ArgumentParser(
        description="Compute free energy of Einstein molecule")
    parser.add_argument('PARAM', type=str,
                        help='json parameter file')
    args = parser.parse_args()

    jdata = json.load(open(args.PARAM))
    fe = free_energy(jdata)
    print('# free energy of Einstein molecule in eV:')
    print(fe)

if __name__ == '__main__' :
    _main()
