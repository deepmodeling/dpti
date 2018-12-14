#!/usr/bin/env python3

import os, re, shutil, logging
import numpy as np
import subprocess as sp

iter_format = "%06d"
task_format = "%02d"
log_iter_head = "iter " + iter_format + " task " + task_format + ": "

def make_iter_name (iter_index) :
    return "iter." + (iter_format % iter_index)

def create_path (path) :
    path += '/'
    if os.path.isdir(path) : 
        dirname = os.path.dirname(path)        
        counter = 0
        while True :
            bk_dirname = dirname + ".bk%03d" % counter
            if not os.path.isdir(bk_dirname) : 
                shutil.move (dirname, bk_dirname) 
                break
            counter += 1
    os.makedirs (path)

def copy_file_list (file_list, from_path, to_path) :
    for jj in file_list : 
        if os.path.isfile(os.path.join(from_path, jj)) :
            shutil.copy (os.path.join(from_path, jj), to_path)
        elif os.path.isdir(os.path.join(from_path, jj)) :
            shutil.copytree (os.path.join(from_path, jj), os.path.join(to_path, jj))


def block_avg(inp, skip = 0, block_size = 10) :
    inp = inp[skip:]
    data_chunks = [
        [j for j in inp[i:i+block_size]] \
        for i in range(0, len(inp), block_size)
    ]
    nblocks = len(data_chunks)
    if (len(data_chunks[-1]) != block_size) :
        nblocks -= 1
        data_chunks = data_chunks[:nblocks]
    assert (len(data_chunks) == nblocks)
    # naive avg
    naive_avg = np.average(inp)
    naive_err = np.std(inp) / np.sqrt(len(inp) - 1)
    # block avg
    data_chunks = np.array(data_chunks)
    data_block = np.average(data_chunks, axis = 1)
    block_avg = np.average(data_block)
    if len(data_block) != 1 :
        block_err = np.std(data_block) / np.sqrt(nblocks-1)
    else :
        block_err = None

    return block_avg, block_err

def cvt_conf (fin, 
              fout, 
              ofmt = 'vasp') :
    """
    Format convert from fin to fout, specify the output format by ofmt
    """
    thisfile = os.path.abspath(__file__)
    thisdir = os.path.dirname(thisfile)
    cmd = os.path.join(thisdir, 'ovito_file_convert.py')
    cmd_opt = '-m '+ofmt
    cmd_line = cmd + ' ' + cmd_opt + ' ' + fin + ' ' + fout
    sp.check_call(cmd_line, shell = True)    
    # sp.check_call([cmd, cmd_opt, fin, fout])

def _parse_one_str(in_s) :
    fmt_s = in_s.split(':') 
    if len(fmt_s) == 1 :
        return np.array([float(fmt_s[0])])
    else :
        assert(len(fmt_s)) == 3 
        return np.arange(float(fmt_s[0]),
                         float(fmt_s[1]), 
                         float(fmt_s[2]))

def parse_seq(in_s) :
    all_l = []
    if type(in_s) == list and type(in_s[0]) == str :
        for ii in in_s :
            for jj in _parse_one_str(ii) :
                all_l.append(jj)      
    elif type(in_s) == list and ( type(in_s[0]) == float or type(in_s[0]) == int ) :
        all_l = [float(ii) for ii in in_s]
    elif type(in_s) == str :
        all_l = _parse_one_str(in_s)
    else :
        raise RuntimeError("the type of seq should be one of: string, list_of_strings, list_of_floats")
    return np.array(all_l)

def integrate(xx, yy) :
    diff_e = 0
    ntasks = len(xx) - 1
    assert (len(yy) - 1 == ntasks)
    for ii in range(ntasks) :
        diff_e += 0.5 * (xx[ii+1] - xx[ii]) * (yy[ii+1] + yy[ii])
    return (diff_e)
    
def integrate(xx, yy, ye) :
    diff_e = 0
    err = 0
    ntasks = len(xx) - 1
    assert (len(yy) - 1 == ntasks)
    for ii in range(ntasks) :
        diff_e += 0.5 * (xx[ii+1] - xx[ii]) * (yy[ii+1] + yy[ii])
        err += np.square(0.5 * (xx[ii+1] - xx[ii]) * ye[ii+1])
        err += np.square(0.5 * (xx[ii+1] - xx[ii]) * ye[ii])
    return diff_e, np.sqrt(err)
    

def _interval_deriv2 (xx, yy) :
    mat = np.ones([3, 3])
    for ii in range(3) :
        mat[ii][0] = xx[ii]**2
        mat[ii][1] = xx[ii]
    coeff = np.linalg.solve(mat, yy)
    return 2.*coeff[0]

def _interval_sys_err (xx, yy, mode) :
    if mode == 'middle' :        
        d0 = np.abs(_interval_deriv2(xx[0:3], yy[0:3]))
        d1 = np.abs(_interval_deriv2(xx[1:4], yy[1:4]))
        dd = np.max([d0, d1])
        dx = np.abs(xx[2] - xx[1])
    elif mode == 'left' :
        dd = np.abs(_interval_deriv2(xx[0:3], yy[0:3]))
        dx = np.abs(xx[1] - xx[0])
    elif mode == 'right' :
        dd = np.abs(_interval_deriv2(xx[0:3], yy[0:3]))
        dx = np.abs(xx[2] - xx[1])
    return 1./12.*(dx**3)*dd


def integrate_sys_err (xx, yy) :
    err = 0
    if len(xx) <= 2 :
        return err
    err += _interval_sys_err(xx[0:3], yy[0:3], 'left')
    # print('here', err)
    for ii in range(1, len(xx)-3) :
        err += _interval_sys_err(xx[ii-1:ii+3], yy[ii-1:ii+3], 'middle')
        # print('here2', _interval_sys_err(xx[ii-1:ii+3], yy[ii-1:ii+3], 'middle'))
    err += _interval_sys_err(xx[-3:], yy[-3:], 'right')    
    # print('here1', _interval_sys_err(xx[-3:], yy[-3:], 'right') )
    return err
