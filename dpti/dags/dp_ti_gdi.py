import json, os
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.subdag_operator import SubDagOperator
from airflow.operators.python import get_current_context
from airflow.decorators import task
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
from airflow.exceptions import AirflowSkipException, DagNotFound, DagRunAlreadyExists, DagRunNotFound, AirflowException,  AirflowFailException
# from airflow.api.client.local_client import Client
from airflow.utils.state import State

from airflow.api.client.local_client import Client
from airflow.models import Variable, DagRun
from numpy.core.fromnumeric import var
# from dpdispatcher.
from dpdispatcher.submission import Submission, Task
from dpdispatcher.batch_object import Machine

import time
import uuid
from dpti import gdi

from dpti.gdi import gdi_main_loop

# default_args = {'owner': 'airflow',
#                 'start_date': datetime(2018, 1, 1)
#                 }

# BASE_DAG_NAME='dag_dpti_gdi_v8'
# MAX_LOOP_NUM = 30

class GDIDAGFactory:
    default_args = {'owner': 'airflow',
            'start_date': datetime(2018, 1, 1)
    }

    dagargs = {
        'default_args': default_args,
        'schedule_interval': None,
    }
    def __init__(self, gdi_name, dag_work_base):
        self.gdi_name = gdi_name
        self.dag_loop_name = self.gdi_name + '_gdi_loop_dag'
        self.dag_main_name = self.gdi_name + '_gdi_main_dag'
        self.var_name = self.gdi_name + '_dv_dh'
        self.dag_work_base = dag_work_base
        self.main_dag = self.create_main_dag()
        self.loop_dag = self.create_loop_dag()

    def create_main_dag(self):
        dag_name = self.dag_loop_name
        var_name = self.var_name
        work_base = self.dag_work_base
        @task()
        def dpti_gdi_main_prepare(**kwargs):
            # context = get_current_context()
            # dag_run = context['params']
            # work_base = dag_run['work_base']
            # Variable.set(self.var_name, 'run')
            prepare_return = True
            return prepare_return
        
        # @task()
        @task(retries=2, retry_delay=timedelta(minutes=1))
        def dpti_gdi_main_loop(prepare_return, **kwargs):
            # context = get_current_context()
            # dag_run = context['params']
            # work_base = dag_run['work_base']
            # work_base = work_base
            print('debug:prepare_return', prepare_return)
            with open(os.path.join(work_base, 'machine.json'), 'r') as f:
                mdata = json.load(f)

            with open(os.path.join(work_base, 'pb.json'), 'r') as f:
                jdata = json.load(f)

            with open(os.path.join(work_base, 'gdidata.json'), 'r') as f:
                gdidata_dict = json.load(f)

            output_dir = os.path.join(work_base, 'new_job/')
            gdidata_dict['output'] = output_dir

            # workflow =
            gdi_workflow = GDIWorkflow(var_name=var_name, 
                dag_name=dag_name)

            gdi_main_loop(jdata=jdata, 
                mdata=mdata,
                gdidata_dict=gdidata_dict,
                gdidata_cli={}, 
                workflow=gdi_workflow
            )
            # return True          
            # Variable.set(self.var_name, 'run')
            loop_return = True
            return loop_return

        @task()
        def dpti_gdi_main_end(loop_return, **kwargs):            
            # Variable.set(self.var_name, 'run')
            end_return = True
            return end_return

        main_dag = DAG(self.dag_main_name, **self.__class__.dagargs)
        with main_dag:
            prepare_return = dpti_gdi_main_prepare()
            loop_return = dpti_gdi_main_loop(prepare_return)
            end_return = dpti_gdi_main_end(loop_return)

        return main_dag

    def create_loop_dag(self):
        @task(multiple_outputs=True)
        def dpti_gdi_loop_prepare():
            # Variable.set(self.var_name, 'run')

            context = get_current_context()
            dag_run = context['params']
            task0_dict = dag_run['task_dict_list'][0]
            task1_dict = dag_run['task_dict_list'][1]

            submission_dict = dag_run['submission_dict']
            # prepare_return = True
            # return (task0_dict, task1_dict)
            return {'task0_dict':task0_dict, 'task1_dict':task1_dict}
            # return (task0_dict, task1_dict)

        # @task()
        @task(retries=2, retry_delay=timedelta(minutes=1))
        def dpti_gdi_loop_md(task_dict):
            context = get_current_context()
            dag_run = context['params']

            submission_dict = dag_run['submission_dict']
            print('submission_dict', submission_dict)
            mdata = dag_run['mdata']
            print('mdata', mdata)
            print('debug:task_dict', task_dict)

            machine = Machine.load_from_machine_dict(mdata)
            batch = machine.batch
            submission = Submission.deserialize(
                submission_dict=submission_dict,
                batch=batch
            )
            submission.register_task(task=Task.deserialize(task_dict=task_dict))
            submission.run_submission()
            # md_return = prepare_return
            return True

        @task()
        def dpti_gdi_loop_end(task0_return, task1_return):
            end_return = True
            # Variable.set(self.var_name, 'end')
            return end_return

        loop_dag = DAG(self.dag_loop_name, **self.__class__.dagargs)
        with loop_dag:
            tasks_dict = dpti_gdi_loop_prepare()
            task0_return = dpti_gdi_loop_md(tasks_dict['task0_dict'])
            task1_return = dpti_gdi_loop_md(tasks_dict['task1_dict'])
            end_return = dpti_gdi_loop_end(task0_return, task1_return)
        return loop_dag


class GDIWorkflow:
    def __init__(self, dag_name, var_name):
        self.dag_name = dag_name
        self.var_name = var_name
        self.run_id = None

    def get_dag_run_state(self):
        if self.run_id is None:
            raise DagRunNotFound(f"dag_id {self.dag_name}; {self.run_id}")

        dag_runs = DagRun.find(
            dag_id=self.dag_name,
            run_id=self.run_id
        )
        
        return dag_runs[0].state if dag_runs else None

    def wait_until_end(self):
        while True:
            dag_run_state = self.get_dag_run_state()
            if dag_run_state == State.SUCCESS:
                print(f"dag_run_state: {dag_run_state}")
                break
            elif dag_run_state == State.RUNNING:
                print(f"dag_run_state: {dag_run_state}")
                time.sleep(30)
            else:
                raise AirflowFailException(f"subdag dag_run fail dag_id:{self.dag_name}; run_id:{self.run_id};")
        return dag_run_state

    def trigger_loop(self, submission, task_list, mdata):
        # loop_num = None
        c = Client(None, None)
        submission_dict = submission.serialize()
        task_dict_list = [task.serialize() for task in task_list]
        submission_hash = submission.submission_hash
        self.run_id = f"dag_run_{submission_hash}"
        try:
            c.trigger_dag(dag_id=self.dag_name, run_id=self.run_id,
                conf={'submission_dict': submission_dict, 'task_dict_list':task_dict_list, 'mdata':mdata}
            ) 
        except DagRunAlreadyExists:
            dag_run_state = self.get_dag_run_state()
            if dag_run_state == State.FAILED:
                raise AirflowFailException(f"subdag dag_run fail dag_id:{self.dag_name}; run_id:{self.run_id};")
            else:
                print(f"continue from old dag_run {self.run_id}")
        loop_return = self.wait_until_end()
        return loop_return





