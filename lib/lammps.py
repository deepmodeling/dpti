#!/usr/bin/env python3

import numpy

def get_thermo(filename) :
    with open(filename, 'r') as fp :
        fc = fp.read().split('\n')
    for sl in range(len(fc)) :
        if 'Step KinEng PotEng TotEng' in fc[sl] :
            break
    for el in range(len(fc)) :
        if 'Loop time of' in fc[el] :
            break
    data = []
    for ii in range(sl+1, el) :
        data.append([float(jj) for jj in fc[ii].split()])
    data = np.array(data)
    return data
