import json, os, sys, glob

# from airflow.models import DAG
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.operators.python import get_current_context
from airflow.operators.bash_operator import BashOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

from airflow.exceptions import AirflowSkipException, AirflowFailException

# from dpdispatcher.lazy_local_context import LazyLocalContext
from dpdispatcher import Submission, Task, Resources, Machine
from dpti import equi, hti, hti_liq, ti
import subprocess as sp

def get_empty_submission(job_work_dir):
    context = get_current_context()
    dag_run = context['params']
    work_base_dir = dag_run['work_base_dir']

    with open(os.path.join(work_base_dir, 'machine.json'), 'r') as f:
        mdata = json.load(f)
    machine = Machine.load_from_dict(mdata['machine'])
    resources = Resources.load_from_dict(mdata['resources'])

    submission = Submission(
        work_base=job_work_dir, 
        resources=resources, 
        machine=machine, 
    )
    return submission

@task()
def all_start_check():
    context = get_current_context()
    print(context)

    dag_run = context['params']
    work_base_dir = dag_run['work_base_dir']
    target_temp = int(dag_run['target_temp'])
    target_pres = int(dag_run['target_pres'])
    conf_lmp = str(dag_run['conf_lmp'])
    ti_path = str(dag_run['ti_path'])
    ens = str(dag_run['ens'])
    if_liquid = dag_run['if_liquid']


    work_base_abs_dir = os.path.realpath(work_base_dir)

    dag_work_dirname=str(target_temp)+'K-'+str(target_pres)+'bar-'+str(conf_lmp)
    dag_work_dir=os.path.join(work_base_abs_dir, dag_work_dirname)

    assert os.path.isdir(work_base_dir) is True,  f'work_base_dir {work_base_dir} must exist '
    if os.path.isdir(dag_work_dir) is False:
        os.mkdir(dag_work_dir)
    else:
        pass

    conf_lmp_abs_path = os.path.join(work_base_abs_dir, conf_lmp)
    assert os.path.isfile(conf_lmp_abs_path) is True,  f'structure file {conf_lmp_abs_path} must exist'
    assert str(ti_path) in ["t", "p"], f'value for "path" must be "t" or "p" '

    start_info = dict(work_base_dir=work_base_dir, 
        target_temp=target_temp,
        target_pres=target_pres, 
        conf_lmp=conf_lmp, 
        ti_path=ti_path,
        ens=ens, 
        if_liquid=if_liquid, 
        work_base_abs_dir=work_base_abs_dir,
        dag_work_dir=dag_work_dir)
    print('start_info:', start_info)
    return start_info

@task()
def NPT_start(start_info):
    print(start_info)
    work_base_abs_dir = start_info['work_base_abs_dir']
    dag_work_dir = start_info['dag_work_dir']
    job_work_dir = os.path.join(dag_work_dir, 'NPT_sim', 'new_job')

    result_json_file = os.path.join(job_work_dir, 'result.json')
    if os.path.isfile(result_json_file):
        return job_work_dir
    with open(os.path.join(work_base_abs_dir, 'npt.json')) as f:
        npt_jdata = json.load(f)

    task_jdata = npt_jdata.copy()
    task_jdata['equi_conf'] = start_info['conf_lmp']
    task_jdata['temp'] = start_info['target_temp']
    task_jdata['pres'] = start_info['target_pres']
    task_jdata['ens'] = start_info['ens']
    print(task_jdata)
    cwd = os.getcwd()
    os.chdir(work_base_abs_dir)
    equi.make_task(job_work_dir, task_jdata)
    os.chdir(cwd)
    return job_work_dir


@task(trigger_rule='none_failed_or_skipped')
def NPT_sim(job_work_dir):
    task = Task(command='lmp -i in.lammps', task_work_path='./', 
        forward_files=['in.lammps', '*lmp', 'graph.pb'], backward_files=['log.lammps'])
    submission = get_empty_submission(job_work_dir)
    submission.register_task_list([task])
    submission.run_submission()
    return job_work_dir

@task(trigger_rule='none_failed')
def NPT_end(job_work_dir):
    result_file_path = os.path.join(job_work_dir, 'result.json')
    if os.path.isfile(result_file_path):
        info = json.load(open(result_file_path, 'r'))
    else:
        info = equi.post_task(job_work_dir)
    print(info)
    return info

@task()
def NVT_start(start_info, *, NPT_end_info):

    print('NPT_end_info', NPT_end_info)
    npt_dir = NPT_end_info['job_dir']
    print('debug', npt_dir)
    
    work_base_abs_dir = start_info.get('work_base_abs_dir')
    dag_work_dir = start_info.get('dag_work_dir')
    job_work_dir = os.path.join(dag_work_dir, 'NVT_sim', 'new_job')

    result_json_file = os.path.join(job_work_dir, 'result.json')
    if os.path.isfile(result_json_file):
        return job_work_dir

    with open(os.path.join(work_base_abs_dir, 'nvt.json')) as f:
        nvt_jdata = json.load(f)

    task_jdata = nvt_jdata.copy()
    task_jdata['temp'] = start_info['target_temp']
    task_jdata['pres'] = start_info['target_pres']
    task_jdata['ens'] = 'nvt' 
    print('debug', npt_dir)
    
    cwd = os.getcwd()
    os.chdir(work_base_abs_dir)
    equi.make_task(job_work_dir, task_jdata, npt_dir=npt_dir)
    os.chdir(cwd)
    return job_work_dir

@task(trigger_rule='none_failed_or_skipped')
def NVT_sim(job_work_dir):
    submission = get_empty_submission(job_work_dir)
    task = Task(command='lmp -i in.lammps', task_work_path='./',
        forward_files=['in.lammps', '*lmp', 'graph.pb'], backward_files=['log.lammps'])
    submission.register_task_list([task])
    submission.run_submission()
    return job_work_dir

@task(trigger_rule='none_failed')
def NVT_end(job_work_dir):
    result_file_path = os.path.join(job_work_dir, 'result.json')
    if os.path.isfile(result_file_path):
        with open(result_file_path, 'r') as f:
            info = json.load(f)
    else:
        info = equi.post_task(job_work_dir)
    print(info)
    return info

@task()
def HTI_start(start_info, *, NVT_end_info={}):
    work_base_abs_dir = start_info['work_base_abs_dir']
    dag_work_dir = start_info['dag_work_dir']
    if_liquid = start_info['if_liquid']
    conf_lmp = NVT_end_info.get('out_lmp', None)

    job_work_dir = os.path.join(dag_work_dir, 'HTI_sim', 'new_job')
    if os.path.isfile(os.path.join(dag_work_dir, 'HTI_sim', 'new_job', 'result.json')):
        return job_work_dir

    if if_liquid:
        with open(os.path.join(work_base_abs_dir, 'hti.liquid.json')) as j:
            hti_jdata = json.load(j)
    else:
        with open(os.path.join(work_base_abs_dir, 'hti.json')) as j:
            hti_jdata = json.load(j)
    
    task_jdata = hti_jdata.copy()
    if conf_lmp is not None:
        task_jdata['equi_conf'] = conf_lmp
    else:
        task_jdata['equi_conf'] = start_info['conf_lmp']

    task_jdata['temp'] = start_info['target_temp']
    task_jdata['pres'] = start_info['target_pres']

    cwd = os.getcwd()
    os.chdir(work_base_abs_dir)
    if if_liquid:
        hti_liq.make_tasks(iter_name=job_work_dir, jdata=task_jdata)
    else:
        hti.make_tasks(iter_name=job_work_dir, jdata=task_jdata, ref='einstein', switch='three-step')
    os.chdir(cwd)
    return job_work_dir

@task()
def HTI_sim(job_work_dir):
    task_abs_dir_list = glob.glob(os.path.join(job_work_dir, './*/task*'))
    task_dir_list = [os.path.relpath(ii, start=job_work_dir ) for ii in task_abs_dir_list]

    task_list = [ Task(command='lmp -i in.lammps', 
        task_work_path=ii, forward_files=['in.lammps', '*lmp', 'graph.pb'], backward_files=['log.lammps']) 
        for ii in task_dir_list ]
    submission = get_empty_submission(job_work_dir)
    # submission.forward_common_files = 
    submission.register_task_list(task_list=task_list)
    submission.run_submission()
    return job_work_dir

@task()
def HTI_end(job_work_dir,
    start_info,
    NPT_end_info={}
):
    if_liquid = start_info['if_liquid']
    manual_pv = NPT_end_info.get('pv')
    manual_pv_err = NPT_end_info.get('pv_err')

    result_file_path = os.path.join(job_work_dir, 'result.json')
    if os.path.isfile(result_file_path):
        info = json.load(open(result_file_path, 'r'))
        return info

    if if_liquid:
        info = hti_liq.compute_task(
            job_work_dir, 
            'gibbs', 
            manual_pv=manual_pv, 
            manual_pv_err=manual_pv_err
        )
    else:
        info = hti.compute_task(
            job_work_dir, 
            'gibbs', 
            manual_pv=manual_pv, 
            manual_pv_err=manual_pv_err
        )
    print(info)
    return info

@task()
def TI_start(start_info, *, HTI_end_info=None):
    work_base_abs_dir = start_info['work_base_abs_dir']
    dag_work_dir = start_info['dag_work_dir']

    ti_path = start_info['ti_path']
    conf_lmp = start_info['conf_lmp']

    if ti_path == 't':
        with open(os.path.join(work_base_abs_dir, 'ti.t.json')) as j:
            ti_jdata = json.load(j)
            task_jdata = ti_jdata.copy()
            task_jdata['pres'] = start_info['target_pres']
            job_dir = 'TI_t_sim'
    elif ti_path == 'p':
        with open(os.path.join(work_base_abs_dir, 'ti.p.json')) as j:
            ti_jdata = json.load(j)
            task_jdata = ti_jdata.copy()
            task_jdata['temp'] = start_info['target_temp']
            job_dir = 'TI_p_sim'
    else:
        raise RuntimeError(f'Error integration path. {ti_path}')

    job_work_dir = os.path.join(dag_work_dir, job_dir, 'new_job')
    task_jdata['ens'] = start_info['ens']
    task_jdata['equi_conf'] = conf_lmp

    cwd = os.getcwd()
    os.chdir(work_base_abs_dir)
    ti.make_tasks(job_work_dir, task_jdata)
    os.chdir(cwd)
    return job_work_dir

@task()
def TI_sim(job_work_dir):
    task_abs_dir_list = glob.glob(os.path.join(job_work_dir, './task*'))
    task_dir_list = [os.path.relpath(ii, start=job_work_dir ) for ii in task_abs_dir_list]
    submission = get_empty_submission(job_work_dir)
    task_list = [ Task(command='lmp -i in.lammps', task_work_path=ii, 
         forward_files=['in.lammps', '*lmp', 'graph.pb'], backward_files=['log.lammps']) for ii in task_dir_list]
    submission.register_task_list(task_list=task_list)
    submission.run_submission()
    return job_work_dir

@task()
def TI_end(job_work_dir, start_info, HTI_end_info):
    Eo = HTI_end_info['e1']
    Eo_err = HTI_end_info['e1_err']

    ti_path = start_info['ti_path']
    if ti_path == 't':
        To = start_info['target_temp']
    if ti_path == 'p':
        To = start_info['target_pres']
    result_file_path = os.path.join(job_work_dir, 'result.json')
    if os.path.isfile(result_file_path):
        with open(result_file_path, 'r') as f:
            info = json.load(f)
    else:
        info = ti.compute_task(job_work_dir, inte_method='inte', Eo=Eo, Eo_err=Eo_err, To=To)
    print(info)
    return info

default_args = {
    'owner': 'fengbo',
    'start_date': datetime(2021, 1, 1, 8, 00)
}

@dag(default_args=default_args, schedule_interval=None,)
def TI_taskflow():
    start_info = all_start_check()
    NPT_end_info = NPT_end(
        NPT_sim(NPT_start(start_info=start_info)))

    NVT_end_info = NVT_end(NVT_sim(NVT_start(
        start_info=start_info, NPT_end_info=NPT_end_info)))

    HTI_end_info = HTI_end(HTI_sim(HTI_start(
            start_info=start_info, 
            NVT_end_info=NVT_end_info)),
        start_info=start_info,
        NPT_end_info=NPT_end_info)

    TI_end_info = TI_end(TI_sim(TI_start(
        start_info=start_info, 
        HTI_end_info=HTI_end_info)), 
        start_info=start_info, 
        HTI_end_info=HTI_end_info)

    return TI_end_info

TI_dag = TI_taskflow()


