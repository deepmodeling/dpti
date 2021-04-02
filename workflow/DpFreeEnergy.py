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

from dpdispatcher.lazy_local_context import LazyLocalContext
from dpdispatcher.submission import Submission, Job, Task, Resources
from dpdispatcher.pbs import PBS
# from functions import NPT_end_func
# sys.path.append("/home/fengbo/deepti-1-yfb-Sn/")
from deepti import equi, hti, hti_liq, ti
import subprocess as sp

def get_dag_work_dir(context):
    dag_run = context['params']
    work_base_dir = dag_run['work_base_dir']
    tar_temp = int(dag_run['tar_temp'])
    tar_press = int(dag_run['tar_press'])
    structure = str(dag_run['structure'])
    path = str(dag_run['path'])

    dag_work_folder=str(tar_temp)+'K-'+str(tar_press)+'bar-'+str(structure)+'-'+str(path)
    work_base_dir = os.path.realpath(work_base_dir)
    dag_work_dir=os.path.join(work_base_dir, dag_work_folder)
    return dag_work_dir
 
@task()
def all_start_check():
    context = get_current_context()
    print(context)
    dag_run = context['params']
    work_base_dir = dag_run['work_base_dir']
    tar_temp = int(dag_run['tar_temp'])
    tar_press = int(dag_run['tar_press'])
    structure = str(dag_run['structure'])
    path = str(dag_run['path'])
    ens = str(dag_run['ens'])
    if_meam = dag_run['if_meam']
    if_liquid = dag_run['if_liquid']
    ti_begin_var = None
    equi_conf = None
    model = None

    work_base_dir = os.path.realpath(work_base_dir)

    dag_work_folder=str(tar_temp)+'K-'+str(tar_press)+'bar-'+str(structure)+'-'+str(path)
    dag_work_dir = get_dag_work_dir(context=get_current_context())
    print(dag_work_dir)
    # dag_work_dir=os.path.join(work_base_dir, dag_work_folder)

    # context['work_base_dir'] = work_base_dir
    # context['dag_work_dir'] = dag_work_dir

    equi_conf = os.path.join(work_base_dir, structure + '.lmp')
    model = os.path.join(work_base_dir, 'graph.pb')

    assert os.path.isdir(work_base_dir) is True,  f'work_base_dir {work_base_dir} must exist '
    # assert os.path.isdir(dag_work_dir) is False,  f'dag_work_folder dir {dag_work_dir} already exist'
    # os.mkdir(dag_work_dir) 
    if os.path.isdir(dag_work_dir) is False:
        os.mkdir(dag_work_dir)
    else:
        pass

    assert os.path.isfile(equi_conf) is True,  f'structure file {equi_conf} must exist'
    assert str(path) in ["t", "p"], f'value for "path" must be "t" or "p" '
    assert type(if_meam) is bool
    if path == "t": ti_begin_var = dag_run['tar_temp'] 
    if path == "p": ti_begin_var = dag_run['tar_press']
    
    start_info = dict(work_base_dir=work_base_dir, tar_temp=tar_temp,
        tar_press=tar_press, structure=structure, path=path, 
         ens=ens, if_meam=if_meam, if_liquid=if_liquid, equi_conf=equi_conf, model=model,
         dag_work_folder=dag_work_folder,
         dag_work_dir=dag_work_dir, ti_begin_var=ti_begin_var)
    return start_info

@task()
def NPT_start(start_info):
    dag_work_dir = get_dag_work_dir(context=get_current_context())
    sim_work_dir = os.path.join(dag_work_dir, 'NPT_sim', 'new_job')
    if os.path.isfile(os.path.join(dag_work_dir, 'NPT_sim', 'new_job', 'result.json')):
        return sim_work_dir
    #     raise AirflowSkipException
    with open(os.path.join(dag_work_dir, '../', 'npt.json')) as j:
        npt_jdata = json.load(j)

    task_jdata = npt_jdata.copy()
    task_jdata['equi_conf'] = start_info['equi_conf']
    task_jdata['model'] = start_info['model']
    task_jdata['temp'] = start_info['tar_temp']
    task_jdata['pres'] = start_info['tar_press']
    task_jdata['ens'] = start_info['ens']
    task_jdata['if_meam'] = start_info['if_meam']
    equi.make_task(sim_work_dir, task_jdata)
    return sim_work_dir

@task(trigger_rule='none_failed_or_skipped')
def NPT_sim(sim_work_dir):
    lazy_local_context = LazyLocalContext(local_root=sim_work_dir, work_profile=None)
    pbs = PBS(context=lazy_local_context)
    resources = Resources(number_node=1, cpu_per_node=4, gpu_per_node=1, 
        queue_name="V100_12_92", group_size=1, if_cuda_multi_devices=False) 
    task = Task(command='lmp_serial -i in.lammps', task_work_path='./') 
    submission = Submission(work_base=sim_work_dir, resources=resources, batch=pbs, task_list=[task])
    submission.run_submission()
    return sim_work_dir

@task(trigger_rule='none_failed')
def NPT_end(sim_work_dir):
    # task_work_dir = start_info['dag']
    result_file_path = os.path.join(sim_work_dir, 'result.json')
    if os.path.isfile(result_file_path):
    # if os.path.isfile(result_file_paath):
        info = json.load(open(result_file_path, 'r'))
    else:
        info = equi.post_task(sim_work_dir)
    print(info)
    return info

@task()
def NVT_start(start_info, *, NPT_end_info=None):
    # print(NVT_start_dict)
 #    work_base_dir = start_info['work_base_dir']
    dag_work_dir = get_dag_work_dir(context=get_current_context())

    sim_work_dir = os.path.join(dag_work_dir, 'NVT_sim', 'new_job')

    if os.path.isfile(os.path.join(dag_work_dir, 'NVT_sim', 'new_job', 'result.json')):
        return sim_work_dir
    #     raise AirflowSkipException

    with open(os.path.join(dag_work_dir, '../', 'nvt.json')) as j:
        nvt_jdata = json.load(j)

    task_jdata = nvt_jdata.copy()
    task_jdata['equi_conf'] = start_info['equi_conf']
    task_jdata['model'] = start_info['model']
    task_jdata['temp'] = start_info['tar_temp']
    task_jdata['pres'] = start_info['tar_press']
    task_jdata['if_meam'] = start_info['if_meam']
 #    assert task_jdata['ens'] == 'nvt'
    # npt_conf = NPT_end_info
    npt_conf = os.path.join(dag_work_dir, 'NPT_sim', 'new_job')
    equi.make_task(sim_work_dir, task_jdata, npt_conf=npt_conf)
    return sim_work_dir
 
@task(trigger_rule='none_failed_or_skipped')
def NVT_sim(sim_work_dir):
    lazy_local_context = LazyLocalContext(local_root=sim_work_dir, work_profile=None)
    pbs = PBS(context=lazy_local_context)
    resources = Resources(number_node=1, cpu_per_node=4, gpu_per_node=1, queue_name="V100_12_92", group_size=1, if_cuda_multi_devices=False) 
    task = Task(command='lmp_serial -i in.lammps', task_work_path='./') 
    submission = Submission(work_base=sim_work_dir, resources=resources, batch=pbs, task_list=[task])
    # submission.register_task(task=task)
    # submission.generate_jobs()
    # submission.bind_batch(batch=pbs)
    submission.run_submission()
    # end_work_dir = sim_work_dir
    return sim_work_dir

@task(trigger_rule='none_failed')
def NVT_end(sim_work_dir):
    result_file_path = os.path.join(sim_work_dir, 'result.json')
    if os.path.isfile(result_file_path):
        info = json.load(open(result_file_path, 'r'))
    else:
        info = equi.post_task(sim_work_dir)
    print(info)
    return info

@task()
def HTI_start(start_info, *, NVT_end_info=None, NPT_end_info=None):
    # work_base_dir = HTI_start_dict['work_base_dir']
    dag_work_dir = start_info['dag_work_dir']
    if_liquid = start_info['if_liquid']

    sim_work_dir = os.path.join(dag_work_dir, 'HTI_sim', 'new_job')
    if os.path.isfile(os.path.join(dag_work_dir, 'HTI_sim', 'new_job', 'result.json')):
        return sim_work_dir

    if if_liquid:
        with open(os.path.join(dag_work_dir, '../','hti.liquid.json')) as j:
            hti_jdata = json.load(j)
    else:
        with open(os.path.join(dag_work_dir, '../','hti.json')) as j:
            hti_jdata = json.load(j)
    
    task_jdata = hti_jdata.copy()
    # task_jdata['equi_conf'] = "../NVT_sim/new_job/out.lmp"
    task_jdata['equi_conf'] = os.path.join(start_info['dag_work_dir'], 'NVT_sim', 'new_job', 'out.lmp')

    task_jdata['model'] = start_info['model']
    task_jdata['temp'] = start_info['tar_temp']
    task_jdata['press'] = start_info['tar_press']
    task_jdata['if_meam'] = start_info['if_meam']
    if if_liquid:
        hti_liq.make_tasks(iter_name=sim_work_dir, jdata=task_jdata, if_meam=None)
    else:
        hti.make_tasks(iter_name=sim_work_dir, jdata=task_jdata, ref='einstein', switch='three-step', if_meam=None)
    return sim_work_dir

@task()
def HTI_sim(sim_work_dir):
    task_dir_real_list = glob.glob(os.path.join(sim_work_dir, './*/task*'))
    task_dir_list = [os.path.relpath(ii, start=sim_work_dir ) for ii in task_dir_real_list]
    lazy_local_context = LazyLocalContext(local_root=sim_work_dir, work_profile=None)
    pbs = PBS(context=lazy_local_context)
    resources = Resources(number_node=1, cpu_per_node=4, gpu_per_node=1, queue_name="V100_12_92", group_size=1, if_cuda_multi_devices=False) 
    task_list = [ Task(command='lmp_serial -i in.lammps', task_work_path=ii) for ii in task_dir_list ]
    submission = Submission(work_base=sim_work_dir, resources=resources, batch=pbs, task_list=task_list)
    # submission.register_task_list(task_list=task_list)
    # submission.generate_jobs()
    # submission.bind_batch(batch=pbs)
    submission.run_submission()
    return sim_work_dir

@task()
def HTI_end(sim_work_dir, start_info, NPT_end_info=None):
    result_file_path = os.path.join(sim_work_dir, 'result.json')
    if_liquid = start_info['if_liquid']
    if os.path.isfile(result_file_path):
        info = json.load(open(result_file_path, 'r'))
    elif NPT_end_info is not None:
        if if_liquid:
            info = hti_liq.compute_task(sim_work_dir, 'gibbs', manual_pv=NPT_end_info['pv'], manual_pv_err=NPT_end_info['pv_err']) 
        else:
            info = hti.compute_task(sim_work_dir, 'gibbs', manual_pv=NPT_end_info['pv'], manual_pv_err=NPT_end_info['pv_err'])
    else:
        if if_liquid:
            info = hti_liq.compute_task(sim_work_dir, 'gibbs')
        else:
            info = hti.compute_task(sim_work_dir, 'gibbs')
    print(info)

    phase_trans_check_result = hti.hti_phase_trans_analyze(sim_work_dir)
    if phase_trans_check_result is True:
        raise AirflowFailException(f"Phase transition happens. {sim_work_dir}")
    return info

@task()
def TI_start(start_info, *, HTI_end_info=None):
    dag_work_dir = start_info['dag_work_dir']

    path = start_info['path']
    if path == 't':
        with open(os.path.join(dag_work_dir, '../','ti.t.json')) as j:
            ti_jdata = json.load(j)
            ti_jdata['press'] = start_info['tar_press']
            sim_dir = 'TI_t_sim'
    elif path == 'p':
        with open(os.path.join(dag_work_dir, '../','ti.p.json')) as j:
            ti_jdata = json.load(j)
            ti_jdata['temps'] = start_info['tar_temp']
            sim_dir = 'TI_p_sim'
    else:
        raise RuntimeError(f'Error integration path. {path}')
    sim_work_dir = os.path.join(dag_work_dir, sim_dir, 'new_job')
    task_jdata = ti_jdata.copy()
    task_jdata['equi_conf'] = os.path.join(start_info['dag_work_dir'], 'NVT_sim', 'new_job', 'out.lmp')
    # task_jdata['equi_conf'] = start_info['equi_conf']
    task_jdata['model'] = start_info['model']
    task_jdata['ens'] = start_info['ens']
    task_jdata['if_meam'] = start_info['if_meam']
    ti.make_tasks(sim_work_dir, task_jdata, if_meam=None)
    return sim_work_dir

@task()
def TI_sim(sim_work_dir):
    task_dir_real_list = glob.glob(os.path.join(sim_work_dir, './task*'))
    task_dir_list = [os.path.relpath(ii, start=sim_work_dir ) for ii in task_dir_real_list]
    lazy_local_context = LazyLocalContext(local_root=sim_work_dir, work_profile=None)
    pbs = PBS(context=lazy_local_context)
    resources = Resources(number_node=1, cpu_per_node=4, gpu_per_node=1, queue_name="V100_12_92", group_size=1, if_cuda_multi_devices=False) 
    submission = Submission(work_base=sim_work_dir, resources=resources)
    task_list = [ Task(command='lmp_serial -i in.lammps', task_work_path=ii) for ii in task_dir_list ]
    submission.register_task_list(task_list=task_list)
    submission.generate_jobs()
    submission.bind_batch(batch=pbs)
    submission.run_submission()
    return sim_work_dir

@task()
def TI_end(sim_work_dir, start_info, HTI_end_info):
    Eo = HTI_end_info['e1']
    Eo_err = HTI_end_info['e1_err']
    ti_jdata = json.load(open(os.path.join(sim_work_dir, 'in.json'), 'r'))
    path = ti_jdata['path']
    if path == 't':
        To = start_info['tar_temp']
    if path == 'p':
        To = start_info['tar_press']
    result_file_path = os.path.join(sim_work_dir, 'result.json')
    if os.path.isfile(result_file_path):
        info = json.load(open(result_file_path, 'r'))
    else:
        info = ti.compute_task(sim_work_dir, inte_method='inte', Eo=Eo, Eo_err=Eo_err, To=To)
    print(info)
    # command = f"/home/fengbo/deepti-1-yfb-Sn/deepti/ti.py compute {sim_work_dir} -t {To} -e {Eo} -E {Eo_err} > {sim_work_dir}/../result"
    # print(command)
    # r = sp.run(["bash", '-l', "-c", command])
    # print(r)
       #  stdout=sp.PIPE,
       #  stderr=sp.STDOUT)
    # os.Popen()
    return info

default_args = {
    'owner': 'fengbo',
    'start_date': datetime(2020, 1, 1, 8, 00)
}


@dag(default_args=default_args, schedule_interval=None, start_date=datetime(2021, 1, 1, 8, 00))
def all_taskflow():
    start_info = all_start_check()
    NPT_end_info = NPT_end(NPT_sim(NPT_start(start_info=start_info)))
    NVT_end_info = NVT_end(NVT_sim(NVT_start(start_info=start_info, NPT_end_info=NPT_end_info)))
    HTI_end_info = HTI_end(HTI_sim(HTI_start(start_info=start_info, NVT_end_info=NVT_end_info)))
    # TI_end_info = TI_


@dag(default_args=default_args, schedule_interval=None, start_date=datetime(2021, 1, 1, 8, 00))
def HTI_taskflow():
    start_info = all_start_check()
    NPT_end_info = NPT_end(NPT_sim(NPT_start(start_info=start_info)))
    NVT_end_info = NVT_end(NVT_sim(NVT_start(start_info=start_info, NPT_end_info=NPT_end_info)))
    HTI_end_info = HTI_end(HTI_sim(HTI_start(start_info=start_info, NVT_end_info=NVT_end_info)), start_info=start_info, NPT_end_info=NPT_end_info)
    
@dag(default_args=default_args, schedule_interval=None, start_date=datetime(2021, 1, 1, 8, 00))
def TI_taskflow():
    start_info = all_start_check()
    NPT_end_info = NPT_end(NPT_sim(NPT_start(start_info=start_info)))
    NVT_end_info = NVT_end(NVT_sim(NVT_start(start_info=start_info, NPT_end_info=NPT_end_info)))
    HTI_end_info = HTI_end(HTI_sim(HTI_start(start_info=start_info, NVT_end_info=NVT_end_info)), start_info=start_info, NPT_end_info=NPT_end_info)
    TI_end_info = TI_end(TI_sim(TI_start(start_info=start_info, HTI_end_info=HTI_end_info)), start_info=start_info, HTI_end_info=HTI_end_info)


    
HTI_dag = HTI_taskflow()
TI_dag = TI_taskflow()


