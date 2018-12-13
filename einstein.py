#!/usr/bin/env python3

import os, sys, json, argparse, glob
import numpy as np
import scipy.constants as pc

from lib.utils import cvt_conf
from lib.vasp import poscar_vol

def compute_lambda(temp, mass) :
    ret = 2. * np.pi * mass * (1e-3 / pc.Avogadro) * pc.Boltzmann * temp / (pc.Planck * pc.Planck)
    return 1./np.sqrt(ret)    

def compute_spring(temp, spring_k) :
    ret = (spring_k * pc.electron_volt / (pc.angstrom * pc.angstrom)) / (pc.Boltzmann * temp * np.pi) 
    return np.sqrt(ret)

def ideal_gas_fe(jdata) :
    equi_conf = jdata['equi_conf']
    temp = jdata['temp']
    mass_map = jdata['model_mass_map']
    tmp_poscar = 'tmp.%06d.POSCAR' % np.random.randint(0, 999999)

    cvt_conf(equi_conf, tmp_poscar)
    vol = poscar_vol(tmp_poscar)
    with open(tmp_poscar) as fp :
        lines = list(fp)
        natoms = [int(ii) for ii in lines[6].split()]    
    os.remove(tmp_poscar)

    Lambda_k = [compute_lambda(temp, ii) for ii in mass_map]    
    fe = 0
    for idx,ii in enumerate(natoms) :
        # kinetic contrib
        rho = ii / (vol * (pc.angstrom**3))
        fe += ii * np.log(rho * (Lambda_k[idx] ** 3)) 
        fe -= ii
        fe += 0.5 * np.log(2. * np.pi * ii)
    fe *= pc.Boltzmann * temp / pc.electron_volt
    return fe

def free_energy (jdata) :
    equi_conf = jdata['equi_conf']
    spring_k = jdata['spring_k']
    temp = jdata['temp']
    mass_map = jdata['model_mass_map']
    tmp_poscar = 'tmp.%06d.POSCAR' % np.random.randint(0, 999999)

    cvt_conf(equi_conf, tmp_poscar)
    vol = poscar_vol(tmp_poscar)
    with open(tmp_poscar) as fp :
        lines = list(fp)
        natoms = [int(ii) for ii in lines[6].split()]    
    os.remove(tmp_poscar)
    
    Lambda_k = [compute_lambda(temp, ii) for ii in mass_map]
    Lambda_s = [compute_spring(temp, ii) for ii in spring_k]
    # print(np.log(Lambda_k), np.log(Lambda_s))
    
    with open(equi_conf) as fp:
        lines = list(fp)
    for idx,ii in enumerate(lines) :
        if 'Atoms #' in ii :
            break
    first_type = int(lines[idx+2].split()[1]) - 1
    print('# fixed atom of type %d ' % first_type)

    fe = 0
    for idx,ii in enumerate(natoms) :
        # kinetic contrib
        fe += 3 * ii * np.log(Lambda_k[idx])
        if (idx == first_type) :            
            fe += 3 * (ii-1) * np.log(Lambda_s[idx])
            fe += np.log(ii / (vol * (pc.angstrom**3)))
        else :
            fe += 3 * ii * np.log(Lambda_s[idx])
    fe *= pc.Boltzmann * temp / pc.electron_volt
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
