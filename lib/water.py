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

def posi_shift(box, r0, r1) :
    rbox = np.linalg.inv(box)
    rbox = rbox.T
    p0 = (np.dot(rbox, r0))
    p1 = (np.dot(rbox, r1))
    dp = p0 - p1
    shift = np.zeros(3)
    for dd in range(3) :
        if dp[dd] >= 0.5 : 
            shift[dd] -= 1
        elif dp[dd] < -0.5 :
            shift[dd] += 1
    return shift

def compute_bonds(box, atype, posis, 
                  max_roh = 1.3, 
                  uniq_hbond = True):
    natoms = len(posis)
    bonds = []
    for ii in range(natoms) :
        bonds.append([])
    for ii in range(natoms) :
        if atype[ii] == 1 :
            for jj in range(natoms) :
                if atype[jj] == 2 :
                    dr = posi_diff(box, posis[ii], posis[jj])
                    if np.linalg.norm(dr) < max_roh :
                        bonds[ii].append(jj)
                        bonds[jj].append(ii)
    if uniq_hbond :
        for jj in range(natoms) :
            if atype[jj] == 2 :
                if len(bonds[jj]) > 1 :
                    orig_bonds = bonds[jj]
                    min_bd = 1e10
                    min_idx = -1
                    for ii in bonds[jj] :
                        dr = posi_diff(box, posis[ii], posis[jj])
                        drr = np.linalg.norm(dr)
                        # print(ii,jj, posis[ii], posis[jj], drr)
                        if drr < min_bd :
                            min_idx = ii
                            min_bd = drr
                    bonds[jj] = [min_idx]
                    orig_bonds.remove(min_idx)
                    # print(min_idx, orig_bonds, bonds[jj])
                    for ii in orig_bonds :
                        bonds[ii].remove(jj)
    return bonds


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
    bonds = compute_bonds(box, atype, posis, max_roh)

    # check water moles
    for ii in range(len(bonds)) :
        if atype[ii] == 1 :
            assert(len(bonds[ii]) == 2), 'ill defined O atom %d has H %s' % (ii, bonds[ii])
        elif atype[ii] == 2 :
            assert(len(bonds[ii]) == 1), 'ill defined H atom %d has O %s' % (ii, bonds[ii]) 

    # pbc posi
    for ii in range(sum(natoms)) :
        if atype[ii] == 1:
            j0idx = bonds[ii][0]
            shift0 = posi_shift(box, posis[j0idx], posis[ii])
            posis[j0idx] = posis[j0idx] + np.dot(box.T, shift0)
            j1idx = bonds[ii][1]
            shift1 = posi_shift(box, posis[j1idx], posis[ii])
            posis[j1idx] = posis[j1idx] + np.dot(box.T, shift1)
    for atoms_idx in range(len(lines)) :
        if 'Atoms' in lines[atoms_idx] :
            break
    for ii in range(sum(natoms)) :
        words = lines[atoms_idx + 2 +ii].split()
        words[2] = str(posis[ii][0])
        words[3] = str(posis[ii][1])
        words[4] = str(posis[ii][2])
        lines[atoms_idx+2+ii] = ' '.join(words)


    ret_bd = []
    idx = 1
    for ii in range(len(bonds)) :
        if atype[ii] == 1:
            ret_bd.append("%d 1 %d %d" % (idx, 1+ii, 1+bonds[ii][0]))
            idx += 1
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


def min_oo (box, atype, posis) :
    natoms = len(posis)
    min_dist = 1e10 * np.ones([natoms])
    for ii in range(natoms) :
        for jj in range(ii+1,natoms) :
            if atype[ii] == 1 and atype[jj] == 1 :
                dij = np.linalg.norm(posi_diff(box, posis[ii], posis[jj]))
                if dij < min_dist[ii] :
                    min_dist[ii] = dij
                if dij < min_dist[jj] :
                    min_dist[jj] = dij
    _min_dist = []
    for ii in min_dist :
        if ii != 1e10 :
            _min_dist.append(ii)
    return _min_dist


def min_ho (box, atype, posis) :
    natoms = len(posis)
    min_dist = 1e10 * np.ones([natoms])
    for ii in range(natoms) :
        for jj in range(natoms) :
            if atype[ii] == 2 and atype[jj] == 1 :
                dij = np.linalg.norm(posi_diff(box, posis[ii], posis[jj]))
                if dij < min_dist[ii] :
                    min_dist[ii] = dij
    _min_dist = []
    for ii in min_dist :
        if ii != 1e10 :
            _min_dist.append(ii)
    return _min_dist        


def min_oho (box, atype, posis) :
    natoms = len(posis)
    dist_oh = []
    dist_oh2 = []
    dist_oo = []
    for ii in range(natoms) :
        if atype[ii] == 1 :
            continue
        all_dist = []
        for jj in range(natoms) :
            if atype[jj] != 1 :
                continue
            dij = np.linalg.norm(posi_diff(box, posis[ii], posis[jj]))
            all_dist.append([dij, jj])
        all_dist.sort()
        doo = np.linalg.norm(posi_diff(box, 
                                       posis[all_dist[0][1]], 
                                       posis[all_dist[1][1]]
        ))
        dist_oh.append(all_dist[0][0])
        dist_oh2.append(all_dist[1][0])
        dist_oo.append(doo)
    # assume O-H..O
    # dist O-H, dist H..O, dist O-O
    return dist_oh, dist_oh2, dist_oo


def min_oh_list(box, atype, posis) :
    natoms = len(posis)
    list_oh = []
    for ii in range(natoms) :
        if atype[ii] == 1 :
            continue
        all_dist = []
        for jj in range(natoms) :
            if atype[jj] != 1 :
                continue
            dij = np.linalg.norm(posi_diff(box, posis[ii], posis[jj]))
            all_dist.append([dij, jj])
        all_dist.sort()
        doo = np.linalg.norm(posi_diff(box, 
                                       posis[all_dist[0][1]], 
                                       posis[all_dist[1][1]]
        ))
        list_oh.append([all_dist[0][1], ii])
    # assume O-H..O
    # dist O-H, dist H..O, dist O-O
    return list_oh


def dist_via_oh_list(box, posis, list_oh) :
    dist = []
    for ii in list_oh :
        dij = np.linalg.norm(posi_diff(box, posis[ii[0]], posis[ii[1]]))
        dist.append(dij)
    return dist


if __name__ == '__main__' :
    fname = 'conf.lmp'
    lines = open(fname).read().split('\n')
    # lines = add_bonds(lines)
    # print('\n'.join(ret_bd))
    # print('\n'.join(ret_ang))

    atype = lmp.get_atype(lines)
    posis = lmp.get_posi(lines)
    bounds, tilt = lmp.get_lmpbox(lines)
    orig, box = lmp.lmpbox2box(bounds, tilt)

    # md_oo = min_oo(box, atype, posis)
    # md_ho = min_ho(box, atype, posis)
    # print(np.average(md_oo), np.average(md_ho), np.min(md_ho), np.max(md_ho))

    moh, moh2, moo = min_oho(box, atype, posis)
    print(np.max(moh), np.average(moh2), np.average(moo))

