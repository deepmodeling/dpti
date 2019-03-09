#!/usr/bin/env python3

import os, sys, json, argparse, glob, shutil
import numpy as np
import scipy.constants as pc
import lib.ti as ti

from lib.utils import create_path
from lib.utils import block_avg

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
        = ti.gen_lammps_input('conf.lmp',
                              model_mass_map, 
                              'graph.pb',
                              nsteps, 
                              dt,
                              ens,
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
    

def make_dpdt (temp, pres, task_path, mdata, ssh_sess) :
    assert(os.path.isdir(task_path))    

    cwd = os.getcwd()
    os.chddir(task_path)

    # check if we need new MD simulations
    new_task = False
    if not os.path.isdir('database') :
        new_task = True
        counter = 0
    else :
        data = np.loadtxt('database/dpdt.out') 
        counter = data.shape[0]
        for ii in data :
            if np.linalg.norm(temp - ii[0]) < 1e4 and \
               np.linalg.norm(pres - ii[1]) < 1e2 :
                new_task = False
                break

    # new MD simulations are needed
    if new_task :
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
        forward_files = ['conf.lmp', 'in.lammps']
        backward_files = ['log.lammps']
        run_tasks = ['0', '1']        
        _group_slurm_jobs(ssh_sess,
                          resources,
                          command,
                          work_path,
                          run_tasks,
                          1,
                          ['graph.pb'],
                          forward_files,
                          backward_files)
        
    os.chdir(cwd)
        
