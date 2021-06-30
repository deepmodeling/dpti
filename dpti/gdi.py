#!/usr/bin/env python3

from operator import sub
import os, sys, json, argparse, glob, shutil, time
from typing import Optional
import numpy as np
import scipy.constants as pc

from scipy.integrate import solve_ivp
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from dpti.lib.utils import create_path, relative_link_file
from dpti.lib.utils import block_avg
from dpti.lib.lammps import get_natoms
from dpti.ti import _gen_lammps_input
from dpti import ti
import dargs
from dargs import dargs, Argument, Variant

# from dpti.workflow
# from lib.RemoteJob import SSHSession, JobStatus, SlurmJob, PBSJob
# from dpgen.dispatcher.Dispatcher import Dispatcher
from dpdispatcher import Submission, Task, Resources, Machine

# try:
#     from airflow.exceptions import AirflowFailException, AirflowSkipException, DagNotFound, DagRunAlreadyExists
#     from airflow.api.client.local_client import Client
# except ImportError:
#     pass

# def _group_slurm_jobs(ssh_sess,
#                       resources,
#                       command,
#                       work_path,
#                       tasks,
#                       group_size,
#                       forward_common_files,
#                       forward_task_files,
#                       backward_task_files,
#                       remote_job = None) :
#     task_chunks = [
#         [os.path.basename(j) for j in tasks[i:i + group_size]] \
#         for i in range(0, len(tasks), group_size)
#     ]
#     job_list = []
#     for chunk in task_chunks :
#         rjob = remote_job(ssh_sess, work_path)
#         rjob.upload('.',  forward_common_files)
#         rjob.upload(chunk, forward_task_files)
#         rjob.submit(chunk, command, resources = resources)
#         job_list.append(rjob)

#     job_fin = [False for ii in job_list]
#     while not all(job_fin) :
#         for idx,rjob in enumerate(job_list) :
#             if not job_fin[idx] :
#                 status = rjob.check_status()
#                 if status == JobStatus.terminated :
#                     raise RuntimeError("find unsuccessfully terminated job in %s" % rjob.get_job_root())
#                 elif status == JobStatus.finished :
#                     rjob.download(task_chunks[idx], backward_task_files)
#                     rjob.clean()
#                     job_fin[idx] = True
#         time.sleep(30)

def _make_tasks_onephase(temp, 
                         pres, 
                         task_path, 
                         jdata,
                         ens = 'npt',
                         conf_file = 'conf.lmp', 
                         graph_file = 'graph.pb',
                         if_meam=False,
                         meam_model=None):
    # assume that model and conf.lmp exist in the current dir
    assert os.path.isfile(conf_file), (conf_file, os.getcwd())
    # assert(os.path.isfile(graph_file))
    conf_file = os.path.abspath(conf_file)
    if graph_file:
        graph_abs_file = os.path.abspath(graph_file)
    
    mass_map = jdata['mass_map']
    # MD simulation protocol
    nsteps = jdata['nsteps']
    timestep = jdata['timestep']
    thermo_freq = jdata['thermo_freq']
    tau_t = jdata['tau_t']
    tau_p = jdata['tau_p']

    cwd = os.getcwd()
    if not os.path.isdir(task_path):
        create_path(task_path)

    os.chdir(task_path)
    if not os.path.exists('conf.lmp'):
        os.symlink(os.path.relpath(conf_file), 'conf.lmp')
    if graph_file:
        if not os.path.exists('graph.pb'):
            os.symlink(os.path.relpath(graph_abs_file), 'graph.pb')

    if if_meam:
        relative_link_file(meam_model['library_abs_path'], './')
        relative_link_file(meam_model['potential_abs_path'], './')
        # meam_library_basename = os.path.basename(meam_model['library'])
        # meam_potential_basename = os.path.basename(meam_model['potential'])
        # os.symlink(os.path.join('../../../', meam_library_basename), meam_library_basename)
        # os.symlink(os.path.join('../../../', meam_potential_basename), meam_potential_basename)
    
    # input for NPT MD
    lmp_str \
        = _gen_lammps_input('conf.lmp',
                               mass_map, 
                               graph_file,
                               nsteps, 
                               timestep,
                               ens,
                               temp,
                               pres,
                               tau_t = tau_t,
                               tau_p = tau_p,
                               thermo_freq = thermo_freq,
                               if_meam=if_meam,
                               meam_model=meam_model)
    with open('thermo.out', 'w') as fp :
        fp.write('%.16e %.16e' % (temp, pres))
    with open('in.lammps', 'w') as fp :
        fp.write(lmp_str)

    os.chdir(cwd)
    # end _make_tasks_onephase


def _setup_dpdt (task_path, jdata) :
    name_0 = jdata['phase_i']['name']
    name_1 = jdata['phase_ii']['name']
    conf_0 = os.path.join(task_path, '../', jdata['phase_i']['equi_conf'])
    conf_1 = os.path.join(task_path, '../', jdata['phase_ii']['equi_conf'])
    conf_0 = os.path.abspath(conf_0)
    conf_1 = os.path.abspath(conf_1)
    model = os.path.join(task_path, '../', jdata['model'])
    if model:
        model = os.path.abspath(model)

    task_abs_dir = create_path(task_path)
    conf_0_name = 'conf.%s.lmp' % '0'
    conf_1_name = 'conf.%s.lmp' % '1'
    # conf_0_name = 'conf.%s.lmp' % name_0
    # conf_1_name = 'conf.%s.lmp' % name_1
    copied_conf_0 = os.path.join(os.path.abspath(task_path), conf_0_name)
    copied_conf_1 = os.path.join(os.path.abspath(task_path), conf_1_name)
    shutil.copyfile(conf_0, copied_conf_0)
    shutil.copyfile(conf_1, copied_conf_1)
    if model:
        linked_model = os.path.join(os.path.abspath(task_path), 'graph.pb')
        shutil.copyfile(model, linked_model)

    with open(os.path.join(os.path.abspath(task_path), 'in.json'), 'w') as fp:
        json.dump(jdata, fp, indent=4)


def make_dpdt (temp,
               pres,
               inte_dir,
               task_path,
               mdata,
               natoms=None,
               shift=[0, 0],
               verbose=False,
               if_meam=False,
               meam_model=None,
               workflow=None):
    assert(os.path.isdir(task_path))    

    if if_meam:
        meam_model['library_abs_path'] = os.path.abspath(meam_model['library'])
        meam_model['potential_abs_path'] = os.path.abspath(meam_model['potential'])
        # relative_link_file(), task_path)
        # relative_link_file(os.path.abspath(meam_model['potential']), task_path)

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

    # try to find nearest simulation
    if new_task and os.path.isfile('database/dpdt.out'):
        data = np.loadtxt('database/dpdt.out')
        data = np.reshape(data, [-1,4])
        min_idx = -1
        min_val = 1e10
        if inte_dir == 't' :
            for ii in range(data.shape[0]) :
                dist = np.abs(data[ii][0] - temp)
                if dist < min_val :
                    min_val = dist
                    min_idx = ii
        elif inte_dir == 'p' :
            for ii in range(data.shape[0]) :
                dist = np.abs(data[ii][1] - pres)
                if dist < min_val :
                    min_val = dist
                    min_idx = ii
        else :
            raise RuntimeError("invalid inte_dir " + inte_dir)
        assert(min_idx >= 0)
        conf_0 = os.path.join('database', 'task.%06d' % min_idx, '0', 'out.lmp')
        conf_1 = os.path.join('database', 'task.%06d' % min_idx, '1', 'out.lmp')
    else :
        conf_0 = 'conf.0.lmp'
        conf_1 = 'conf.1.lmp'

    # new MD simulations are needed
    if new_task :
        if verbose :
            print('# dpdt: do not find any matched record, run new task from %d ' % counter)

        with open('in.json', 'r') as j:
            jdata = json.load(j)
        # make new task
        work_base = os.path.abspath(os.path.join('database', 'task.%06d' % counter))

        _make_tasks_onephase(temp, pres, 
                             os.path.join(work_base, '0'),
                             jdata,
                             ens = jdata['phase_i'].get('ens', None),
                             conf_file = conf_0,
                             graph_file = 'graph.pb',
                             if_meam=if_meam,
                             meam_model=meam_model)
        _make_tasks_onephase(temp, pres, 
                             os.path.join(work_base, '1'),
                             jdata, 
                             ens = jdata['phase_ii'].get('ens', None),
                             conf_file = conf_1,
                             graph_file = 'graph.pb',
                             if_meam=if_meam,
                             meam_model=meam_model)
        # submit new task

        # if workflow is None:
        machine = Machine.load_from_dict(mdata['machine'])
        resources = Resources.load_from_dict(mdata['resources'])

        command = 'lmp -i in.lammps'
        forward_files = ['conf.lmp', 'in.lammps', 'graph.pb']
        if if_meam:
            meam_library_basename = os.path.basename(meam_model['library'])
            meam_potential_basename = os.path.basename(meam_model['potential'])
            forward_files.extend([meam_library_basename, meam_potential_basename])
        backward_files = ['log.lammps', 'out.lmp']

        task_list = []
        for ii in range(2):
            task = Task(
                command=command,
                task_work_path=f'{ii}/',
                forward_files=forward_files,
                backward_files=backward_files
            )
            task_list.append(task)

        submission = Submission(
            work_base=work_base,
            machine=machine,
            resources=resources,
            forward_common_files=[],
            backward_common_files=[],
        )
        if workflow is None:
            submission.register_task_list(task_list=task_list)
            # submission.generate_jobs()
            submission.run_submission()
        else:
        # client.trigg
            workflow.trigger_loop(submission=submission, task_list=task_list, mdata=mdata)

        # run_tasks = ['0', '1']        

        # dispatcher.run_jobs(resources,
        #                     command,
        #                     work_path,
        #                     run_tasks,
        #                     1,
        #                     [],
        #                     forward_files,
        #                     backward_files,
        #                     forward_task_deference = False)
        # _group_slurm_jobs(ssh_sess,
        #                   resources,
        #                   command,
        #                   work_path,
        #                   run_tasks,
        #                   1,
        #                   [],
        #                   forward_files,
        #                   backward_files)

        # collect resutls
        log_0 = os.path.join(work_base, '0', 'log.lammps')
        log_1 = os.path.join(work_base, '1', 'log.lammps')
        if natoms == None :
            natoms = [get_natoms('conf.0.lmp'), get_natoms('conf.1.lmp')]
        stat_skip = jdata['stat_skip']
        stat_bsize = jdata['stat_bsize']
        t0 = ti._compute_thermo(log_0, natoms[0], stat_skip, stat_bsize)
        t1 = ti._compute_thermo(log_1, natoms[1], stat_skip, stat_bsize)
        dv = t1['v'] - t0['v']
        dh = t1['h'] - t0['h'] - (shift[1] - shift[0])
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
                  shift = [0, 0],
                  verbose = False,
                  if_meam=False,
                  meam_model=None,
                  workflow=None
    ):
        self.jdata = jdata
        self.mdata = mdata
        self.task_path = task_path
        self.inte_dir = inte_dir
        self.natoms = natoms
        self.verbose =  verbose
        self.pref = pref
        self.shift = shift
        self.if_meam = if_meam
        self.meam_model = meam_model
        self.workflow = workflow

        # self.dispatcher = Dispatcher(mdata['machine'], context_type = 'lazy-local', batch_type = 'pbs')
        if os.path.isdir(task_path) :
            print('find path ' + task_path + ' use it. The user should guarantee the consistency between the jdata and the found work path ')
        else :
            _setup_dpdt(task_path, jdata)

        self.ev2bar = pc.electron_volt / (pc.angstrom ** 3) * 1e-5

    def __call__ (self, x, y) :
        if self.inte_dir == 't' :
            # x: temp, y: pres
            [dv, dh] = make_dpdt(x, y,
                                 self.inte_dir,
                                 self.task_path, self.mdata,
                                 self.natoms,
                                 self.shift,
                                 self.verbose,
                                 if_meam=self.if_meam,
                                 meam_model=self.meam_model,
                                 workflow=self.workflow)
            return [dh / (x * dv) * self.ev2bar * self.pref]

        elif self.inte_dir == 'p' :
            # x: pres, y: temp
            [dv, dh] = make_dpdt(y, x,
                                 self.inte_dir,
                                 self.task_path, self.mdata,
                                 self.natoms,
                                 self.shift,
                                 self.verbose,
                                 if_meam=self.if_meam,
                                 meam_model=self.meam_model,
                                 workflow=self.workflow)
            return [(y * dv) / dh / self.ev2bar * (1/self.pref)]

# def gdi_main_loop(jdata, mdata, gdidata, begin=None, end=None, direction=None,
#     initial_value=None, step_value=None, abs_tol=10, rel_tol=0.01, if_water=None,
#     output=None, first_step=None, shift=[0.0, 0.0], verbose=None, if_meam=None, workflow=None):

def gdi_main_loop(jdata, mdata, gdidata_dict, gdidata_cli={}, workflow=None):

    gdiargs = [
        Argument("begin", float, optional=False),
        Argument("end", float, optional=False),
        Argument("direction", str, optional=False),
        Argument("initial_value", float, optional=False),
        Argument("step_value", [list, float], optional=True, default=None),
        Argument("abs_tol", float, optional=True, default=10,),
        Argument("rel_tol", float, optional=True, default=0.01),
        Argument("if_water", bool, optional=True, default=None),
        Argument("output", str, optional=True, default="new_job/"),
        Argument("first_step", float, optional=True, default=None),
        Argument("shift", list, optional=True, default=[0.0, 0.0]),
        Argument("verbose", bool, optional=True, default=True),
        Argument("if_meam", bool, optional=True, default=None),
    ]

    gdidata_format = Argument("gdidata", dict, gdiargs)

    gdidata = gdidata_dict.copy()
    gdidata = gdidata_format.normalize_value(gdidata_cli)
    gdidata = gdidata_format.normalize_value(gdidata_dict)

    with open(os.path.join(os.path.dirname(os.path.abspath(gdidata['output'])), 
            'gdidata.run.json'), 'w') as f:
        json.dump(gdidata, f, indent=4)

    natoms = None
    if gdidata['if_water']:
        conf_0 = jdata['phase_i']['equi_conf']
        conf_1 = jdata['phase_ii']['equi_conf']
        natoms = [get_natoms(conf_0), get_natoms(conf_1)]
        natoms = [ii // 3  for ii in natoms]
    print ('# natoms: ', natoms)
    print ('# shifts: ', gdidata['shift'])
    meam_model = jdata.get('meam_model', None)

    # with open('')

    gdf = GibbsDuhemFunc(jdata,
                        mdata,
                        task_path=gdidata['output'],
                        inte_dir=gdidata['direction'],
                        natoms = natoms,
                        shift = gdidata['shift'],
                        verbose = gdidata['verbose'],
                        if_meam=gdidata['if_meam'],
                        meam_model=meam_model,
                        workflow=workflow)

    sol = solve_ivp(gdf,
                    [gdidata['begin'], gdidata['end']],
                    [gdidata['initial_value']],
                    t_eval = gdidata['step_value'],
                    method = 'RK23',
                    atol=gdidata['abs_tol'],
                    rtol=gdidata['rel_tol'],
                    first_step = gdidata['first_step']
                    )

    if gdidata['direction'] == 't' :
        tmp = np.concatenate([sol.t, sol.y[0]])
    else :
        tmp = np.concatenate([sol.y[0], sol.t])        

    tmp = np.reshape(tmp, [2,-1])
    np.savetxt(os.path.join(gdidata['output'], 'pb.out'), tmp.T)
    return True

def _main () :
    parser = argparse.ArgumentParser(
        description="Compute the phase boundary via Gibbs-Duhem integration")
    parser.add_argument('PARAM', type=str,
                        help='json parameter file')
    parser.add_argument('MACHINE', type=str,
                        help='json machine file')
    parser.add_argument('-g', '--gdidata-json', type=str, default=None,
                        help='json gdi integration file')
    parser.add_argument('-b','--begin', type=float, default=None,
                        help='start of the integration')
    parser.add_argument('-e','--end', type=float, default=None,
                        help='end of the integration')
    parser.add_argument('-d','--direction', type=str, choices=['t','p'], default=None,
                        help='direction of the integration, along T or P')
    parser.add_argument('-i','--initial-value', type=float, default=None,
                        help='the initial value of T (direction=p) or P (direction=t)')
    parser.add_argument('-s','--step-value', type=float, nargs = '+',
                        help='the T (direction=t) or P (direction=p) values must be evaluated')
    parser.add_argument('-a','--abs-tol', type=float, default = 10,
                        help='the absolute tolerance of the integration')
    parser.add_argument('-r','--rel-tol', type=float, default = 1e-2,
                        help='the relative tolerance of the integration')
    parser.add_argument('-w','--if-water', action = 'store_true',
                        help='assumes water molecules: nmols = natoms//3')
    parser.add_argument('-o','--output', type=str, default = 'new_job',
                        help='the output folder for the job')
    parser.add_argument('-f','--first-step', type=float, default=None,
                        help='the first step size of the integrator')
    parser.add_argument('-S','--shift', type=float, nargs = 2, default = [0.0, 0.0],
                        help='the output folder for the job')
    parser.add_argument('-v','--verbose', action = 'store_true',
                        help='print detailed infomation')
    parser.add_argument("-z", "--if-meam", help="whether use meam instead of dp", action="store_true")
    args = parser.parse_args()

    with open(args.PARAM) as j:
        jdata = json.load(j)
    with open(args.MACHINE) as m:
        mdata = json.load(m)

    if args.gdidata_json:
        with open(args.gdidata_json) as g:
            gdidata_dict = json.load(g)
    else:
        gdidata_dict=None

    gdidata_cli = dict(begin=args.begin, end=args.end,
        direction=args.direction, initial_value=args.initial_value, step_value=args.step_value, 
        abs_tol=args.abs_tol, rel_tol=args.rel_tol, if_water=args.if_water, output=args.output,
        first_step=args.first_step, shift=args.shift, verbose=args.verbose, if_meam=args.if_meam
    )

    return_value = gdi_main_loop(jdata=jdata, 
        mdata=mdata,
        gdidata_dict=gdidata_dict,
        gdidata_cli=gdidata_cli,
        workflow=None)
    # gdidata_run = gdiargs



    # return_value = gdi_main_loop(jdata=jdata, mdata=mdata, gdidata=gdidata, begin=args.begin, end=args.end,
    #     direction=args.direction, initial_value=args.initial_value, step_value=args.step_value, 
    #     abs_tol=args.abs_tol, rel_tol=args.rel_tol, if_water=args.if_water, output=args.output,
    #     first_step=args.first_step, shift=args.shift, verbose=args.verbose, if_meam=args.if_meam, workflow=None)

    return return_value

if __name__ == '__main__' :
    _main()        
    
# mdata = json.load(open('machine.json'))
# jdata = json.load(open('in.json'))
# # ssh_sess = SSHSession(mdata['machine'])
# # if not os.path.isdir('gdi_test') :
# #     _setup_dpdt('gdi_test', jdata)
# # make_dpdt(100, 1,  'gdi_test', mdata, ssh_sess, natoms = [96, 128])
# # make_dpdt(100, 20, 'gdi_test', mdata, ssh_sess, natoms = [96, 128])

# gdf = GibbsDuhemFunc(jdata, mdata, 'gdi_test', 'p', natoms = [96, 128], verbose = True)

# sol = solve_ivp(gdf, [1, 20000], [363], method = 'RK23', atol=10, rtol=1e-2)
# print(sol.t)
# print(sol.y)
# np.savetxt('t.out', sol.t)
# np.savetxt('p.out', sol.y)
# # print(gdf(100, 1))
# # print(gdf(100, 1))
# # print(gdf(200, 20))
