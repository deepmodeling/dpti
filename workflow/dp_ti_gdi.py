from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.subdag_operator import SubDagOperator
from airflow.decorators import task
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
import random

default_args = {'owner': 'airflow',
                'start_date': datetime(2018, 1, 1)
                }

dag = DAG('dag_dpti_gdi1',
            schedule_interval=None,
            default_args=default_args)

@task()
def gdi_loop_start(ii, **kwargs):
    print("loop_start_check")
    return 0

@task()
def gdi_loop_calculate(ii, **kwargs):
    return_value = ii + 1
    print("return_value:", return_value)
    return return_value

@task()
def gdi_loop_end(ii, **kwargs):
    print("return_value:", ii)
    return ii

with dag:
    return_value_list = []
    return_value_list.append(gdi_loop_start(0))

    n = 10
    for ii in range(n):
        new_return_value = gdi_loop_calculate(return_value_list[ii])
        return_value_list.append(new_return_value)
    end_return_value = gdi_loop_end(return_value_list[n])



