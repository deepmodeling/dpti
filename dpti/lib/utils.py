#!/usr/bin/env python3

import os, re, shutil, logging
import numpy as np
import subprocess as sp
import hashlib, pathlib

iter_format = "%06d"
task_format = "%02d"
log_iter_head = "iter " + iter_format + " task " + task_format + ": "
float_protect = 1e-14

def make_iter_name (iter_index) :
    return "iter." + (iter_format % iter_index)


def get_first_matched_key_from_dict(dct, lst):
    value = None
    for key in lst:
        if key in dct:
            value = dct[key]
            return value
    raise KeyError(f"not found key in dct={dct}, lst={lst}")


def create_dict_not_empty_key(**kwargs):
    dct = {}
    for k,v in kwargs.items():
        if v is not None:
            dct[k] = v
    return dct

# def create_path (path) :
#     path += '/'
#     if os.path.isdir(path) : 
#         dirname = os.path.dirname(path)        
#         counter = 0
#         while True :
#             bk_dirname = dirname + ".bk%03d" % counter
#             if not os.path.isdir(bk_dirname) : 
#                 shutil.move (dirname, bk_dirname) 
#                 break
#             counter += 1
#     os.makedirs (path)





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
    abs_path = os.path.abspath(path)
    return abs_path


def relative_link_file(file_path, target_dir):
    if not os.path.isfile(file_path):
        raise RuntimeError(f"file_path:{file_path} must be a file. cwd:{os.getcwd()}")
    file_abs_path = os.path.abspath(file_path)
    basename = os.path.basename(file_abs_path)
    relative_path = os.path.relpath(file_abs_path, start=target_dir)
    target_linkfile_path = os.path.join(target_dir, basename)
    os.symlink(src=relative_path, dst=target_linkfile_path)
    return target_linkfile_path

def link_file_in_dict(dct, 
        key_list, 
        target_dir
    ):
    if not dct:
        return {}
    return_dict = {}
    for k in key_list:
        file_path = dct.get(k, None)
        if file_path is not None:
            target_linkfile_path = relative_link_file(
                file_path=file_path,
                target_dir=target_dir
            )
            v = os.path.basename(target_linkfile_path)
            return_dict[k] = v
    return return_dict

def get_file_md5(file_path):
    return hashlib.md5(pathlib.Path(file_path).read_bytes()).hexdigest()

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
        # print(float(fmt_s[0]), float(fmt_s[1]), float(fmt_s[2]))
        return np.arange(float(fmt_s[0]),
                         float(fmt_s[1]) - float_protect, 
                         float(fmt_s[2]))

def parse_seq(in_s, *, protect_eps=None):
    all_l = []
    if type(in_s) == list and type(in_s[0]) == str :
        for ii in in_s :
            for jj in _parse_one_str(ii) :
                all_l.append(jj)      
    elif type(in_s) == list and \
         ( type(in_s[0]) == float or \
           type(in_s[0]) == np.float32 or \
           type(in_s[0]) == np.float64 or \
           type(in_s[0]) == int ) :
        all_l = [float(ii) for ii in in_s]
    elif type(in_s) == str :
        all_l = _parse_one_str(in_s)
    else :
        raise RuntimeError("the type of seq should be one of: string, list_of_strings, list_of_floats")
    if protect_eps is not None:
        if all_l[0] == 0 :
            all_l[0] += protect_eps
        if all_l[-1] == 1 :
            all_l[-1] -= protect_eps
    # return all_l
    return np.array(all_l)

# def integrate(xx, yy) :
#     diff_e = 0
#     ntasks = len(xx) - 1
#     assert (len(yy) - 1 == ntasks)
#     for ii in range(ntasks) :
#         diff_e += 0.5 * (xx[ii+1] - xx[ii]) * (yy[ii+1] + yy[ii])
#     return (diff_e)
    
def integrate_trapezoidal(xx, yy, ye) :
    diff_e = 0
    err = 0
    ntasks = len(xx) - 1
    assert (len(yy) - 1 == ntasks)
    for ii in range(ntasks) :
        diff_e += 0.5 * (xx[ii+1] - xx[ii]) * (yy[ii+1] + yy[ii])
        err += np.square(0.5 * (xx[ii+1] - xx[ii]) * ye[ii+1])
        err += np.square(0.5 * (xx[ii+1] - xx[ii]) * ye[ii])
    return diff_e, np.sqrt(err)

def integrate_simpson(xx, yy, ye):
    if len(xx) % 2 == 0:
        diff_e, err = integrate_simpson(xx[:-1], yy[:-1], ye[:-1])
        diff_e1, err1 = integrate_trapezoidal(xx[-2:], yy[-2:], ye[-2:])
        return diff_e+diff_e1, np.linalg.norm([err, err1])
    else:
        diff_e = 0
        err = 0
        for ii in range(0, len(xx)-2, 2):
            two_h = (xx[ii+2] - xx[ii])
            diff_e += 1./6. * two_h * (yy[ii] + 4.*yy[ii+1] + yy[ii+2])
            err += np.square(1./6. * two_h * 1. * ye[ii])
            err += np.square(1./6. * two_h * 4. * ye[ii+1])
            err += np.square(1./6. * two_h * 1. * ye[ii+2])
        return diff_e, np.sqrt(err)

def integrate_simpson_nonuniform(x, f, fe):
    N = len(x) - 1
    h = np.diff(x)
    result = 0.0
    error = 0.0
    for i in range(1, N, 2):
        hph = h[i] + h[i-1]
        pref1 = ( h[i]**3 + h[i-1]**3 + 3. * h[i] * h[i-1] * hph ) / ( 6 * h[i] * h[i-1] )
        pref0 = ( 2. * h[i-1]**3 - h[i]**3 + 3. * h[i] * h[i-1]**2) / ( 6 * h[i-1] * hph)
        pref2 = ( 2. * h[i]**3 - h[i-1]**3 + 3. * h[i-1] * h[i]**2) / ( 6 * h[i] * hph )
        result += f[i  ] * pref1
        result += f[i-1] * pref0
        result += f[i+1] * pref2
        error += np.square(fe[i  ] * pref1)
        error += np.square(fe[i-1] * pref0)
        error += np.square(fe[i+1] * pref2)
    if (N + 1) % 2 == 0:
        pref1 = ( 2 * h[N-1]**2 + 3. * h[N-2] * h[N-1]) / ( 6 * ( h[N-2] + h[N-1] ) )
        pref0 = ( h[N-1]**2 + 3*h[N-1]* h[N-2] ) / ( 6 * h[N-2] )
        pref2 = h[N-1]**3 / ( 6 * h[N-2] * ( h[N-2] + h[N-1] ) )
        result += f[N  ] * pref1
        result += f[N-1] * pref0
        result -= f[N-2] * pref2
        error += np.square(fe[N  ] * pref1)
        error += np.square(fe[N-1] * pref0)
        error += np.square(fe[N-2] * pref2)
    return result, np.sqrt(error)

def integrate(xx, yy, ye, scheme_ = 's'):
    scheme = (scheme_.lower()[0])
    if scheme == 't':
        return integrate_trapezoidal(xx, yy, ye)
    elif scheme == 's':
        return integrate_simpson_nonuniform(xx, yy, ye)
    else:
        raise RuntimeError('unknow integration scheme', scheme_)

def _interval_deriv2 (xx, yy) :
    mat = np.ones([3, 3])
    for ii in range(3) :
        mat[ii][0] = xx[ii]**2
        mat[ii][1] = xx[ii]
    coeff = np.linalg.solve(mat, yy)
    return 2.*coeff[0]

def interval_sys_err_trapezoidal (xx, yy, mode) :
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


def integrate_sys_err_trapezoidal (xx, yy) :
    err = 0
    if len(xx) <= 2 :
        return err
    err += interval_sys_err_trapezoidal(xx[0:3], yy[0:3], 'left')
    # print('here', err)
    for ii in range(1, len(xx)-2) :
        err += interval_sys_err_trapezoidal(xx[ii-1:ii+3], yy[ii-1:ii+3], 'middle')
        # print('here2', interval_sys_err(xx[ii-1:ii+3], yy[ii-1:ii+3], 'middle'))
    err += interval_sys_err_trapezoidal(xx[-3:], yy[-3:], 'right')    
    # print('here1', interval_sys_err(xx[-3:], yy[-3:], 'right') )
    return err


def integrate_sys_err_simpson (xx, yy) :
    err = 0
    if len(xx) <= 4 :
        return err
    xx = np.array(xx)
    ye = np.zeros(xx.size)
    interval_error = np.zeros(xx.size//4+1)
    for ii in range(xx.size//4):
        inte0,_ = integrate_simpson_nonuniform(xx[ii*4:ii*4+5], yy[ii*4:ii*4+5], ye[ii*4:ii*4+5])
        inte1,_ = integrate_simpson_nonuniform(xx[ii*4:ii*4+5:2], yy[ii*4:ii*4+5:2], ye[ii*4:ii*4+5:2])
        err = np.abs(inte0 - inte1)
        interval_error[ii+1] = err
    # from scipy.interpolate import interp1d
    # err = interp1d(xx[::4], err0, kind='cubic')
    return interval_error[-1]


def integrate_sys_err (xx, yy, scheme_ = 's') :
    scheme = (scheme_.lower()[0])
    if scheme == 't':
        return integrate_sys_err_trapezoidal(xx, yy)
    elif scheme == 's':
        return integrate_sys_err_simpson(xx, yy)
    else:
        raise RuntimeError('unknow integration scheme', scheme_)

def integrate_range_trapezoidal(xx, yy, ye):
    xx = np.array(xx)
    nn = xx.size
    inte = np.zeros([nn])
    stat_err = np.zeros([nn])
    inte_err = np.zeros([nn])
    # stat_err[0] = ye[0]
    for ii in range(1, nn):
        inter_i, inter_se = integrate_trapezoidal(xx[ii-1:ii+1], yy[ii-1:ii+1], ye[ii-1:ii+1])
        inte[ii] = inte[ii-1] + inter_i
        stat_err[ii] = np.sqrt(stat_err[ii-1]**2 + inter_se**2)
    if nn <= 2:
        return xx, inte, inte_err, stat_err
    inte_err[1] = interval_sys_err_trapezoidal(xx[0:3], yy[0:3], 'left')
    for ii in range(1, nn-2):
        inte_err[ii+1] = inte_err[ii] + interval_sys_err_trapezoidal(xx[ii-1:ii+3], yy[ii-1:ii+3], 'middle')
    inte_err[nn-1] = inte_err[nn-2] + interval_sys_err_trapezoidal(xx[-3:], yy[-3:], 'right')
    return xx, inte, inte_err, stat_err

def _integrate_range_simpson_inner(xx, yy, ye):
    xx = np.array(xx)
    nn = xx.size
    new_xx = [xx[0]]
    inte = [0]
    # stat_err = [ye[0]]
    stat_err = [0]
    inte_err = [0]
    for ii in range(2, nn, 2):
        inter_i, inter_se = integrate_simpson_nonuniform(xx[ii-2:ii+1], yy[ii-2:ii+1], ye[ii-2:ii+1])
        new_xx.append(xx[ii])
        inte.append(inte[-1] + inter_i)
        stat_err.append(np.sqrt(stat_err[-1]**2 + inter_se**2))
    return np.array(new_xx), np.array(inte), np.array(stat_err)

def integrate_range_simpson(xx, yy, ye):
    xx = np.array(xx)
    # error esti series 0
    xx0, inte0, stat_err0 = _integrate_range_simpson_inner(xx, yy, ye)
    if len(xx) < 5:
        return xx0, inte0, stat_err0, np.zeros(xx0.shape)
    xx1, inte1, stat_err1 = _integrate_range_simpson_inner(xx[::2], yy[::2], ye[::2])
    diff1 = np.abs(inte1-inte0[::2]) / 16.0
    assert(np.linalg.norm(xx1-xx0[::2]) < 1e-10)
    # error esti series 1, shifted from series 0
    xx2, inte2, stat_err2 = _integrate_range_simpson_inner(xx[2:], yy[2:], ye[2:])
    xx3, inte3, stat_err3 = _integrate_range_simpson_inner(xx[2::2], yy[2::2], ye[2::2])
    from scipy.interpolate import interp1d
    f = interp1d(xx1, diff1)
    diff3 = np.abs(inte3 - inte2[::2]) / 16.0 + f(xx2[0])
    assert(np.linalg.norm(xx3-xx2[::2]) < 1e-10)
    # combine the estimates
    inte_err0 = np.zeros(xx0.shape)
    for ii in range(inte_err0.size):
        if ii % 2 == 0:
            inte_err0[ii] = diff1[ii//2]
        else:
            inte_err0[ii] = diff3[ii//2]    
    return xx0, inte0, inte_err0, stat_err0

def integrate_range (xx, yy, ye, scheme = 's') :
    scheme_ = (scheme.lower()[0])
    if scheme_ == 't':
        return integrate_range_trapezoidal(xx, yy, ye)
    elif scheme_ == 's':
        return integrate_range_simpson(xx, yy, ye)
    else:
        raise RuntimeError('unknow integration scheme', scheme)


def compute_nrefine (all_t, integrand, err, error_scale = None) :
    ntask = all_t.size
    interval_err = []
    interval_err.append(interval_sys_err_trapezoidal(all_t[0:3], integrand[0:3], 'left'))
    for ii in range(1, ntask-2) :
        interval_err.append(
            interval_sys_err_trapezoidal(all_t[ii-1:ii+3], integrand[ii-1:ii+3], 'middle'))
    interval_err.append(interval_sys_err_trapezoidal(all_t[-3:], integrand[-3:], 'right'))
    if error_scale is not None :
        for ii in range(0, ntask-1) :
            interval_err[ii] *= error_scale[ii+1]
    
    interval_nrefine = []
    for ii in range(ntask-1) :
        err_dist = err * (all_t[ii+1] - all_t[ii]) / (all_t[-1] - all_t[0])
        interval_nrefine.append(max(1, int(np.ceil(np.sqrt(interval_err[ii] / err_dist)))))
    #print(interval_nrefine)
    assert(len(interval_nrefine) == len(interval_err))

    return interval_nrefine


def get_task_file_abspath(task_name, file_name): 
    equi_conf = file_name
    cwd = os.getcwd()
    os.chdir(task_name)
    equi_conf = os.path.abspath(equi_conf)
    os.chdir(cwd)
    return equi_conf

def integrate_range_hti(all_lambda, de, de_err, scheme='s'):
    new_lambda, i, i_e, s_e = integrate_range(all_lambda, de, de_err, scheme='s')
    # print('debug:range_hti', new_lambda[-1], all_lambda[-1])
    if new_lambda[-1] != all_lambda[-1] :
        if new_lambda[-1] == all_lambda[-2]:
            _, i1, i_e1, s_e1 = integrate_range(all_lambda[-2:], de[-2:], de_err[-2:], scheme='t')
            diff_e = i[-1] + i1[-1]
            stt_err = np.linalg.norm([s_e[-1], s_e1[-1]])
            sys_err = i_e[-1] + i_e1[-1]
        else :
            raise RuntimeError("lambda does not match!")
    else:
        diff_e = i[-1]
        stt_err = s_e[-1]
        sys_err = i_e[-1]
    return diff_e, stt_err, sys_err


if __name__ == '__main__':
    ninter = 20
    error = 1e-1
    xx = np.arange(0, 1+1e-10, 1./ninter)
    yy = np.exp(xx)
    ye = np.ones(xx.shape) * error / np.sqrt(12)
    yy = yy + (np.random.random([ninter+1]) * error - 0.5 * error)
    print('here', np.random.random([ninter+1]) * error - 0.5 * error)
    # diff0, err0 = integrate_simpson(xx, yy, ye)
    # diff1, err1 = integrate_simpson_nonuniform(xx, yy, ye)
    # real_err0 = np.abs(np.exp(1) - np.exp(0) - diff0)
    # real_err1 = np.abs(np.exp(1) - np.exp(0) - diff1)
    # print('real_error %.2e\nstat_error %.2e' % (real_err0, err0))
    # print('real_error %.2e\nstat_error %.2e' % (real_err1, err1))

    xx1, ii1, i_err1, s_err1 = integrate_range_simpson(xx, yy, ye)
    real_err1 = np.abs(ii1 - (np.exp(xx1) - np.exp(0)))
    print(real_err1)
    print(i_err1)
    print(s_err1)

