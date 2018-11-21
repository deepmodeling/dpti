#!/usr/bin/env python3

import os,sys
import numpy as np
lib_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append (lib_path)
import lmp as lmp


def posi_diff(box, r0, r1) :
    rbox = np.linalg.inv(box)
    rbox = rbox.T
    p0 = (np.dot(rbox, r0))
    p1 = (np.dot(rbox, r1))
    dp = p0 - p1
    shift = np.zeros(3)
    for dd in range(3) :
        if dp[dd] >= 0.5 : 
            dp[dd] -= 1
        elif dp[dd] < -0.5 :
            dp[dd] += 1
    dr = np.dot(box.T, dp)    
    return dr


def add_bonds (lines_, max_roh = 1.3) :
    lines = lines_
    natoms = lmp.get_natoms_vec(lines)
    assert(len(natoms) == 2) 
    # type 1 == O, type 2 == H
    assert(natoms[0] == natoms[1] // 2) 
    
    atype = lmp.get_atype(lines)
    posis = lmp.get_posi(lines)
    bounds, tilt = lmp.get_lmpbox(lines)
    orig, box = lmp.lmpbox2box(bounds, tilt)
    posi_diff(box, posis[0], posis[1])

    bonds = []
    for ii in range(sum(natoms)) :
        bonds.append([])
    for ii in range(sum(natoms)) :
        if atype[ii] == 1 :
            for jj in range(sum(natoms)) :
                if atype[jj] == 2 :
                    dr = posi_diff(box, posis[ii], posis[jj])
                    if np.linalg.norm(dr) < max_roh :
                        bonds[ii].append(jj)
                        bonds[jj].append(ii)

    # check water moles
    for ii in range(len(bonds)) :
        if atype[ii] == 1 :
            assert(len(bonds[ii]) == 2), 'ill defined O atom %d has H %s' % (ii, bonds[ii])
        elif atype[ii] == 2 :
            assert(len(bonds[ii]) == 1), 'ill defined H atom %d has O %s' % (ii, bonds[ii]) 

    ret_bd = []
    idx = 1
    for ii in range(len(bonds)) :
        if atype[ii] == 1:
            ret_bd.append("%d 1 %d %d" % (idx, 1+ii, 1+bonds[ii][0]))
            ret_bd.append("%d 1 %d %d" % (idx, 1+ii, 1+bonds[ii][1]))
            idx += 1

    ret_ang = []
    idx = 1
    for ii in range(len(bonds)) :
        if atype[ii] == 1:
            ret_ang.append("%d 1 %d %d %d" % (idx, 1+bonds[ii][0], 1+ii, 1+bonds[ii][1]))
            idx += 1

    lines.append('Bonds')
    lines.append('')
    lines += ret_bd
    lines.append('')
    lines.append('Angles')
    lines.append('')
    lines += ret_ang
    lines.append('')
    lines.append('')

    nbonds = len(ret_bd)
    nangles = len(ret_ang)
    for atoms_idx in range(len(lines)) :
        if 'atoms' in lines[atoms_idx] :
            break
    lines.insert(atoms_idx+1, '%d angles' % nangles)
    lines.insert(atoms_idx+1, '%d bonds' % nbonds)
    for atoms_types_idx in range(len(lines)) :
        if 'atom types' in lines[atoms_types_idx] :
            break
    lines.insert(atoms_types_idx+1, '%d angle types' % 1)
    lines.insert(atoms_types_idx+1, '%d bond types' % 1)

    for atoms_idx in range(len(lines)) :
        if 'Atoms' in lines[atoms_idx] :
            break
    mole_idx = np.zeros(sum(natoms), dtype = int)
    cc = 1
    for ii in range(sum(natoms)) :
        if atype[ii] == 1 :
            mole_idx[ii] = cc
            cc += 1
    for ii in range(sum(natoms)) :
        if atype[ii] == 2 :
            mole_idx[ii] = mole_idx[bonds[ii]]
    cc = 0
    for idx in range(atoms_idx+2, atoms_idx+2+sum(natoms)) :
        words = lines[idx].split()
        words.insert(1, '%d' % mole_idx[cc])
        lines[idx] = ' '.join(words)
        cc += 1

    return lines


if __name__ == '__main__' :
    fname = 'ice0c.1e0.lmp'
    lines = open(fname).read().split('\n')
    lines = add_bonds(lines)
    # print('\n'.join(ret_bd))
    # print('\n'.join(ret_ang))

    open('tmp.lmp', 'w').write('\n'.join(lines))
