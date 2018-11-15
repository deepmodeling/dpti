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

