#!/usr/bin/python3 

import warnings
import numpy as np

def regulate_poscar(poscar_in, poscar_out) :
    with open(poscar_in, 'r') as fp:
        lines = fp.read().split('\n')
    names = lines[5].split()
    counts = [int(ii) for ii in lines[6].split()]
    assert(len(names) == len(counts))
    uniq_name = []
    for ii in names :
        if not (ii in uniq_name) :
            uniq_name.append(ii)
    uniq_count = np.zeros(len(uniq_name), dtype = int)
    for nn,cc in zip(names,counts) :
        uniq_count[uniq_name.index(nn)] += cc
    natoms = np.sum(uniq_count)
    posis = lines[8:8+natoms]
    all_lines = []
    for ele in uniq_name:
        ele_lines = []
        for ii in posis :
            ele_name = ii.split()[-1]
            if ele_name == ele :
                ele_lines.append(ii) 
        all_lines += ele_lines
    all_lines.append('')
    ret = lines[0:5]
    ret.append(" ".join(uniq_name))
    ret.append(" ".join([str(ii) for ii in uniq_count]))
    ret.append("Direct")
    ret += all_lines
    with open(poscar_out, 'w') as fp:
        fp.write("\n".join(ret))

def sort_poscar(poscar_in, poscar_out, new_names) :
    with open(poscar_in, 'r') as fp:
        lines = fp.read().split('\n')
    names = lines[5].split()
    counts = [int(ii) for ii in lines[6].split()]
    new_counts = np.zeros(len(counts), dtype = int)
    for nn,cc in zip(names,counts) :
        new_counts[new_names.index(nn)] += cc
    natoms = np.sum(new_counts)
    posis = lines[8:8+natoms]
    all_lines = []
    for ele in new_names:
        ele_lines = []
        for ii in posis :
            ele_name = ii.split()[-1]
            if ele_name == ele :
                ele_lines.append(ii) 
        all_lines += ele_lines
    all_lines.append('')
    ret = lines[0:5]
    ret.append(" ".join(new_names))
    ret.append(" ".join([str(ii) for ii in new_counts]))
    ret.append("Direct")
    ret += all_lines
    with open(poscar_out, 'w') as fp:
        fp.write("\n".join(ret))    

def perturb_xz (poscar_in, poscar_out, pert = 0.01) :
    with open(poscar_in, 'r') as fp:
        lines = fp.read().split('\n')
    zz = lines[4]
    az = [float(ii) for ii in zz.split()]
    az[0] += pert
    zz = [str(ii) for ii in az]
    zz = " ".join(zz)
    lines[4] = zz
    with open(poscar_out, 'w') as fp:
        fp.write("\n".join(lines))

def reciprocal_box(box) :
    rbox = np.linalg.inv(box)
    rbox = rbox.T
    # rbox = rbox / np.linalg.det(box)
    # print(np.matmul(box, rbox.T))
    # print(rbox)
    return rbox

def _poscar_natoms(lines) :
    numb_atoms = 0
    for ii in lines[6].split() :
        numb_atoms += int(ii)
    return numb_atoms

def _poscar_scale_direct (str_in, scale) :
    lines = str_in.copy()
    numb_atoms = _poscar_natoms(lines)
    pscale = float(lines[1])
    pscale = pscale * scale
    lines[1] = str(pscale) + "\n"
    return lines

def _poscar_scale_cartesian (str_in, scale) :
    lines = str_in.copy()
    numb_atoms = _poscar_natoms(lines)
    # scale box
    for ii in range(2,5) :
        boxl = lines[ii].split()
        boxv = [float(ii) for ii in boxl]
        boxv = np.array(boxv) * scale
        lines[ii] = "%.16e %.16e %.16e\n" % (boxv[0], boxv[1], boxv[2])
    # scale coord
    for ii in range(8, 8+numb_atoms) :
        cl = lines[ii].split()
        cv = [float(ii) for ii in cl]
        cv = np.array(cv) * scale
        lines[ii] = "%.16e %.16e %.16e\n" % (cv[0], cv[1], cv[2])
    return lines    

def poscar_natoms(poscar_in) :
    with open(poscar_in, 'r') as fin :
        lines = list(fin)
    return _poscar_natoms(lines)

def poscar_scale (poscar_in, poscar_out, scale) :
    with open(poscar_in, 'r') as fin :
        lines = list(fin)
    if 'D' == lines[7][0] or 'd' == lines[7][0] : 
        lines = _poscar_scale_direct(lines, scale)
    elif 'C' == lines[7][0] or 'c' == lines[7][0] : 
        lines = _poscar_scale_cartesian(lines, scale)
    else :
        raise RuntimeError("Unknow poscar coord style at line 7: %s" % lines[7])
    with open(poscar_out, 'w') as fout:
        fout.write("".join(lines))

def poscar_vol (poscar_in) :
    with open(poscar_in, 'r') as fin :
        lines = list(fin)
    box = []
    for ii in range(2,5) :
        words = lines[ii].split()
        vec = [float(jj) for jj in words]
        box.append(vec)
    scale = float(lines[1].split()[0])
    box = np.array(box)
    box *= scale
    return np.linalg.det(box)

