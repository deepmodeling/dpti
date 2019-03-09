#!/usr/bin/env python3

import os, sys, json, argparse, glob, shutil, time
import numpy as np
import scipy.constants as pc
import ti

from lib.utils import create_path
from lib.utils import block_avg
from lib.RemoteJob import SSHSession, JobStatus, SlurmJob

def _group_slurm_jobs(ssh_sess,
                      resources,
                      command,
                      work_path,
                      tasks,
                      group_size,
                      forward_common_files,
                      forward_task_files,
                      backward_task_files,
                      remote_job = SlurmJob) :
    task_chunks = [
        [os.path.basename(j) for j in tasks[i:i + group_size]] \
        for i in range(0, len(tasks), group_size)
    ]
    job_list = []
    for chunk in task_chunks :
        rjob = remote_job(ssh_sess, work_path)
        rjob.upload('.',  forward_common_files)
        rjob.upload(chunk, forward_task_files)
        rjob.submit(chunk, command, resources = resources)
        job_list.append(rjob)

    job_fin = [False for ii in job_list]
    while not all(job_fin) :
        for idx,rjob in enumerate(job_list) :
            if not job_fin[idx] :
                status = rjob.check_status()
                if status == JobStatus.terminated :
                    raise RuntimeError("find unsuccessfully terminated job in %s" % rjob.get_job_root())
                elif status == JobStatus.finished :
                    rjob.download(task_chunks[idx], backward_task_files)
                    rjob.clean()
                    job_fin[idx] = True
        time.sleep(10)

def _make_tasks_onephase(temp, 
                         pres, 
                         task_path, 
                         jdata, 
                         conf_file = 'conf.lmp', 
                         graph_file = 'graph.pb') :
    # assume that model and conf.lmp exist in the current dir
    assert(os.path.isfile(conf_file))
    assert(os.path.isfile(graph_file))
    conf_file = os.path.abspath(conf_file)
    graph_file = os.path.abspath(graph_file)
    model_mass_map = jdata['model_mass_map']
    # MD simulation protocol
    nsteps = jdata['nsteps']
    dt = jdata['dt']
    stat_freq = jdata['stat_freq']
    tau_t = jdata['tau_t']
    tau_p = jdata['tau_p']

    cwd = os.getcwd()
    create_path(task_path)
    os.chdir(task_path)
    os.symlink(os.path.relpath(conf_file), 'conf.lmp')
    os.symlink(os.path.relpath(graph_file), 'graph.pb')
    
    # input for NPT MD
    lmp_str \
        = ti._gen_lammps_input('conf.lmp',
                               model_mass_map, 
                               'graph.pb',
                               nsteps, 
                               dt,
                               'npt',
                               temp,
                               pres,
                               tau_t = tau_t,
                               tau_p = tau_p,
                               prt_freq = stat_freq)
    with open('thermo.out', 'w') as fp :
        fp.write('%.16e %.16e' % (temp, pres))
    with open('in.lammps', 'w') as fp :
        fp.write(lmp_str)

    os.chdir(cwd)
    # end _make_tasks_onephase


def _setup_dpdt (task_path, jdata) :
    name_0 = jdata['phase_i']['name']
    name_1 = jdata['phase_ii']['name']
    conf_0 = jdata['phase_i']['equi_conf']
    conf_1 = jdata['phase_ii']['equi_conf']
    conf_0 = os.path.abspath(conf_0)
    conf_1 = os.path.abspath(conf_1)
    model = jdata['model']
    model = os.path.abspath(model)

    create_path(task_path)
    conf_0_name = 'conf.%s.lmp' % '0'
    conf_1_name = 'conf.%s.lmp' % '1'
    # conf_0_name = 'conf.%s.lmp' % name_0
    # conf_1_name = 'conf.%s.lmp' % name_1
    copied_conf_0 = os.path.join(os.path.abspath(task_path), conf_0_name)
    copied_conf_1 = os.path.join(os.path.abspath(task_path), conf_1_name)
    shutil.copyfile(conf_0, copied_conf_0)
    shutil.copyfile(conf_1, copied_conf_1)
    linked_model = os.path.join(os.path.abspath(task_path), 'graph.pb')
    shutil.copyfile(model, linked_model)

    with open(os.path.join(os.path.abspath(task_path), 'in.json'), 'w') as fp:
        json.dump(jdata, fp, indent=4)
    

def make_dpdt (temp,
               pres,
               task_path,
               mdata,
               ssh_sess,
               natoms = None,
               verbose = False) :
    assert(os.path.isdir(task_path))    

    cwd = os.getcwd()
    os.chdir(task_path)

    # check if we need new MD simulations
    new_task = True
    if (not os.path.isdir('database')) or \
       (not os.path.isfile('database/dpdt.out')):
        if verbose :
            print('# dpdt: cannot find any MD record, start from scrtach')
        new_task = True
        counter = 0
    else :
        if verbose :
            print('# dpdt: found MD records, search if any record matches')
        data = np.loadtxt('database/dpdt.out')
        data = np.reshape(data, [-1,4])
        counter = data.shape[0]
        for ii in range(data.shape[0]) :
            if (np.linalg.norm(temp - data[ii][0]) < 1e-4) and \
               (np.linalg.norm(pres - data[ii][1]) < 1e-2) :
                if verbose :
                    print('# dpdt: found matched record at %f %f ' % (temp, pres))
                new_task = False
                dv = data[ii][2]
                dh = data[ii][3]
                break

    # new MD simulations are needed
    if new_task :
        if verbose :
            print('# dpdt: do not find any matched record, run new task from %d ' % counter)
        jdata = json.load(open('in.json', 'r'))        
        # make new task
        work_path = os.path.join('database', 'task.%06d' % counter)
        _make_tasks_onephase(temp, pres, 
                             os.path.join(work_path, '0'),
                             jdata, 
                             conf_file = 'conf.0.lmp', 
                             graph_file = 'graph.pb')
        _make_tasks_onephase(temp, pres, 
                             os.path.join(work_path, '1'),
                             jdata, 
                             conf_file = 'conf.1.lmp', 
                             graph_file = 'graph.pb')
        # submit new task
        resources = mdata['resources']
        lmp_exec = mdata['lmp_command']
        command = lmp_exec + " -i in.lammps > /dev/null"
        forward_files = ['conf.lmp', 'in.lammps', 'graph.pb']
        backward_files = ['log.lammps']
        run_tasks = ['0', '1']        
        _group_slurm_jobs(ssh_sess,
                          resources,
                          command,
                          work_path,
                          run_tasks,
                          1,
                          [],
                          forward_files,
                          backward_files)
        # collect resutls
        log_0 = os.path.join(work_path, '0', 'log.lammps')
        log_1 = os.path.join(work_path, '1', 'log.lammps')
        if natoms == None :
            natoms = [get_natoms('conf.0.lmp'), get_natoms('conf.1.lmp')]
        stat_skip = jdata['stat_skip']
        stat_bsize = jdata['stat_bsize']
        t0 = ti._compute_thermo(log_0, natoms[0], stat_skip, stat_bsize)
        t1 = ti._compute_thermo(log_1, natoms[1], stat_skip, stat_bsize)
        dv = t1['v'] - t0['v']
        dh = t1['h'] - t0['h']
        with open(os.path.join('database', 'dpdt.out'), 'a') as fp:
            fp.write('%.16e %.16e %.16e %.16e\n' % \
                     (temp, pres, dv, dh))            
    os.chdir(cwd)
    return [dv, dh]


class GibbsDuhemFunc (object):
    def __init__ (self,
                  jdata,
                  mdata,
                  task_path,
                  inte_dir,
                  pref = 1.0,
                  natoms = None,
                  verbose = False):
        self.jdata = jdata
        self.mdata = mdata
        self.task_path = task_path
        self.inte_dir = inte_dir
        self.natoms = natoms
        self.verbose =  verbose
        self.pref = pref
        
        self.ssh_sess = SSHSession(mdata['machine'])
        if os.path.isdir(task_path) :
            print('find path ' + task_path + ' use it. The user should guarantee the consistency between the jdata and the found work path ')
        else :
            _setup_dpdt(task_path, jdata)

        self.ev2bar = pc.electron_volt / (pc.angstrom ** 3) * 1e-5

    def __call__ (self, x, y) :
        if self.inte_dir == 't' :
            # x: temp, y: pres
            [dv, dh] = make_dpdt(x, y,
                                 self.task_path, self.mdata, self.ssh_sess, self.natoms, self.verbose)
            return dh / (x * dv) * self.ev2bar * self.pref
        elif self.inte_dir == 'p' :
            # x: pres, y: temp
            [dv, dh] = make_dpdt(y, x,
                                 self.task_path, self.mdata, self.ssh_sess, self.natoms, self.verbose)
            return (y * dv) / dh / self.ev2bar * (1/self.pref)

mdata = json.load(open('machine.json'))
jdata = json.load(open('in.json'))
# ssh_sess = SSHSession(mdata['machine'])
# if not os.path.isdir('gdi_test') :
#     _setup_dpdt('gdi_test', jdata)
# make_dpdt(100, 1,  'gdi_test', mdata, ssh_sess, natoms = [96, 128])
# make_dpdt(100, 20, 'gdi_test', mdata, ssh_sess, natoms = [96, 128])

gdf = GibbsDuhemFunc(jdata, mdata, 'gdi_test', 't', natoms = [96, 128], verbose = True)
print(gdf(100, 1))
print(gdf(100, 1))
print(gdf(200, 20))
