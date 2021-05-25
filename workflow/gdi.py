from airflow import DAG
from dpti.dags.dp_ti_gdi import GDIDAGFactory

GDI_dag_factory = GDIDAGFactory(gdi_name='Sn_gdi_270K_test31', 
    dag_work_base='/home/fengbo/4_Sn/16_gdi_3/test_Sn.p.270K.1GPa-2GPa')
loop_dag = GDI_dag_factory.loop_dag
main_dag = GDI_dag_factory.main_dag


