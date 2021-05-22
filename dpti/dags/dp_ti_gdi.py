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

import time

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
        self.dag_name = self.gdi_name + '_gdi_dag'
        self.var_name = self.gdi_name + '_dv_dh'
        self.dag = self.create_dag()
        # self.loop_number = 8

    # @property
    # def dag(self):
    #     pass

    def get_dags(self):
        main_dag = self.create_main()
        # loop_dag = self.create_loop()
        return main_dag

    #  @classmethod
    def create_dag(self):
        @task()
        def dpti_gdi_loop_prepare(**kwargs):
            prepare_return = 0
            return prepare_return

        @task()
        def dpti_gdi_loop_md(prepare_return, **kwargs):
            md_return = prepare_return
            return md_return

        @task()
        def dpti_gdi_loop_end(md_return, **kwargs):
            end_return = md_return
            # Variable.set
            var_name = self.var_name
            Variable.set(var_name, end_return)
            return end_return

        dagname = self.gdi_name + '_dpti_gdi'
        dag = DAG(dagname, **self.__class__.dagargs)
        with dag:
            prepare_return = dpti_gdi_loop_prepare()
            md_return = dpti_gdi_loop_md(prepare_return)
            end_return = dpti_gdi_loop_end(md_return)
            print("end_return", end_return)
            # t1 = DummyOperator(task_id='dpti_gdi_loop_prepare')
            # t2 = DummyOperator(task_id='dpti_gdi_loop_md')
            # t3 = DummyOperator(task_id='dpti_gdi_loop_end')
            # t1 >> t2  >> t3
        return dag

    # def get_loop_end_return(self):
    #     var_name = self.var_name
    #     loop_end_return = Variable.get(var_name)
    #     return loop_end_return

    def wait_until_end(self):
        var_begin_value = Variable.get(self.var_name)
        var_value = None
        print('wait until end; var_begin_value', var_begin_value)
        while True:
            var_value = Variable.get(self.var_name)
            if var_value == var_begin_value:
                time.sleep(20)
            else:
                break
        return var_value

    def trigger_loop(self, loop_num):
        # loop_num = None
        c = Client(None, None)
        c.trigger_dag(dag_id=self.dag_name, run_id=f"loop_{loop_num}") #, conf={'loop_num': loop_num})
        loop_return = self.wait_until_end()
        # loop_return = get_loop_end_return()
        return loop_return

    # def 

# GDI_dag_factory = GDIDAGFactory(gdi_name='Sn_test1')
# dag = GDI_dag_factory.main_dag





