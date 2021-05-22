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
from airflow.exceptions import AirflowFailException, AirflowSkipException
# from airflow.api.client.local_client import Client

from airflow.api.client.local_client import Client
from airflow.models import Variable
from numpy.core.fromnumeric import var
# from dpdispatcher.
from dpdispatcher.submission import Submission
from dpdispatcher.batch_object import Machine

import time
from dpti import gdi

from dpti.gdi import gdi_main_loop

default_args = {'owner': 'airflow',
                'start_date': datetime(2018, 1, 1)
                }

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
    def __init__(self, gdi_name):
        self.gdi_name = gdi_name
        self.dag_loop_name = self.gdi_name + '_gdi_loop_dag'
        self.dag_main_name = self.gdi_name + '_gdi_main_dag'
        self.var_name = self.gdi_name + '_dv_dh'
        self.loop_dag = self.create_loop_dag()

    def run_main_loop(self, work_base):
        with open(os.path.join(work_base, 'machine.json'), 'r') as f:
            mdata = json.load(f)

        with open(os.path.join(work_base, 'pb.json'), 'r') as f:
            jdata = json.load(f)

        with open(os.path.join(work_base, 'gdidata.json'), 'r') as f:
            gdidata = json.load(f)

        output_dir = os.path.join(work_base, 'new_job')

        gdi_main_loop(jdata=jdata, 
            mdata=mdata, 
            gdidata=gdidata, 
            output=output_dir, 
            workflow=self.loop_dag
        )
        return True

    def create_loop_dag(self):
        @task()
        def dpti_gdi_loop_prepare(**kwargs):
            Variable.set(self.var_name, False)
            prepare_return = True
            return prepare_return

        @task()
        def dpti_gdi_loop_md(prepare_return, **kwargs):
            context = get_current_context()
            dag_run = context['params']

            submission_dict = dag_run['submission_dict']
            mdata = dag_run['mdata']

            machine = Machine.load_from_machine_dict(mdata)
            batch = machine.batch
            submission = Submission.deserialize(
                submission_dict=submission_dict,
                batch=batch
            )
            submission.run_submission()
            # md_return = prepare_return
            return True

        @task()
        def dpti_gdi_loop_end(md_return, **kwargs):
            end_return = True
            Variable.set(self.var_name, True)
            return end_return

        dag = DAG(self.dag_loop_name, **self.__class__.dagargs)
        with dag:
            prepare_return = dpti_gdi_loop_prepare()
            md_return = dpti_gdi_loop_md(prepare_return)
            end_return = dpti_gdi_loop_end(md_return)
            print("end_return", end_return)
        return dag

    def wait_until_end(self):
        var_begin_value = Variable.get(self.var_name)
        var_value = None
        print('wait until end; var_begin_value', var_begin_value)
        while True:
            var_value = Variable.get(self.var_name)
            if var_value is False:
                time.sleep(20)
            else:
                break
        return var_value

    def trigger_loop(self, submission, mdata):
        # loop_num = None
        c = Client(None, None)
        submission_hash = submission.submission_hash
        c.trigger_dag(dag_id=self.dag_loop_name, run_id=f"gdi_{submission_hash}",
            conf={'submission_dict': submission.serialize(), 'mdata':mdata}
        ) #, conf={'loop_num': loop_num})
        loop_return = self.wait_until_end()
        # loop_return = get_loop_end_return()
        return loop_return


# GDI_dag_factory = GDIDAGFactory(gdi_name='Sn_test1')
# dag = GDI_dag_factory.main_dag




