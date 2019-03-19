#!/usr/bin/env python3

import numpy as np
from lib.utils import integrate
from lib.utils import integrate_sys_err
from lib.lammps import get_thermo
from lib.dump import split_traj
from lib.dump import get_posi
from lib.dump import get_atype
import lib.water as water
import lib.dump as dump

# def func (xx) :
#     return 0.02*xx*xx+0.01*xx+0.03

# x0 = np.arange(0,10.1)
# x1 = np.arange(0,10.1, 0.5)
# x2 = np.arange(0,10.1, 0.25)


# i0 = integrate(x0, func(x0), np.zeros(x0.shape))
# i1 = integrate(x1, func(x1), np.zeros(x1.shape))
# i2 = integrate(x2, func(x2), np.zeros(x2.shape))
# e0 = integrate_sys_err(x0, func(x0))
# e1 = integrate_sys_err(x1, func(x1))
# e2 = integrate_sys_err(x2, func(x2))

# print(i0[0], e0)
# print(i1[0], e1)
# print(i2[0], e2)

# get_thermo('log.lammps')

lines = open('dump.hti').read().split('\n')
ret = split_traj(lines) 
# print(get_posi(ret[0]))
# print(get_posi(ret[0])[127:130])
# print(get_atype(ret[0]))
# print(get_atype(ret[0])[127:130])

bd, tl = dump.get_dumpbox(ret[0])
orig, box = dump.dumpbox2box(bd, tl)
atype = dump.get_atype(ret[0])
posi = dump.get_posi(ret[0])
oh_list = water.min_oh_list(box, atype, posi)

for idx, ii in enumerate(ret) :
    bd, tl = dump.get_dumpbox(ii)
    orig, box = dump.dumpbox2box(bd, tl)
    posi = dump.get_posi(ii)
    dists = water.dist_via_oh_list(box, posi, oh_list)
    print(idx, np.min(dists), np.max(dists), np.average(dists))
          
