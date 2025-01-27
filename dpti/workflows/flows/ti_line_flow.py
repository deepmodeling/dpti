
import os
import json
from typing import NamedTuple, Optional 
from pydantic import AliasChoices, BaseModel, Field, ValidationError
from dpti.workflows.service.workflow_service_module import WorkflowServiceModule, WorkflowService

from injector import provider, Injector, inject, singleton, Module
from prefect import flow, task

from prefect.artifacts import create_link_artifact, create_markdown_artifact

#%%
from ..service.workflow_service_module import WorkflowServiceModule, WorkflowService
from ..service.di import InjectionContext, context_inject, injection_context

from ..simulations.equi_sim import NPTEquiSimulation, NVTEquiSimulation, NPTResultToNVTConfLmp, ExtractNVTToHTIConfLmp
from ..simulations.hti_sim import HTISimulation, HTIResultData
from ..simulations.ti_sim import TISimulation
from ..prefect_task_hash import task_input_json_hash
#%%



class ThermoInputData(NamedTuple):
    equi_conf: str
    pres: float
    temp: float
    ens: str

# @task(cache_key_fn=task_input_hash, persist_result=True)
@task(cache_key_fn=task_input_json_hash, persist_result=True, refresh_cache=True)
def FreeEnergyLineWorkflowStart():
    # free_energy_line_dict = {}

    # pwd = os.getcwd()
    # flow_running_dir = os.path.realpath(flow_trigger_dir)

    thermo_input = ThermoInputData(
        equi_conf="beta.lmp",
        pres=30000,
        temp=300,
        ens='npt-xy'
    )
    print(f"note: FreeEnergyLineWorkflowStar {thermo_input=}")
    return thermo_input

#%%


class Configuration:
    def __init__(self, file_handler_string):
        self.file_handler_string = file_handler_string

local_file_config = Configuration(':localfile:')

def configure_for_testing(binder):
    configuration = local_file_config
    # binder.bind(Configuration, to=configuration, scope=singleton)
    binder.bind(Configuration, to=configuration)


#%%


class FreeEnergyFlow(BaseModel):
    conf_lmp: str
    target_temp: int
    target_pres: int
    work_base_dir: str
    ti_path: str
    ens: str
    if_liquid: bool


#%%



markdown_report = """This flow return `info`:  {r}
![Logo Image](https://github.com/deepmodeling/deepmd-kit/raw/r2/doc/_static/logo.svg)
result: /home/felix/1_software/dpti/examples/NPT_sim/new_job/result
"""

#%%

@flow(log_prints=True, persist_result=True)
def FreeEnergyLineWorkflow(config_json: str, flow_trigger_dir: str, refresh_cache:Optional[bool]=None):
    print(f"!!!NOTE by dpti developer: the results with charts and pictures can be view at tab:results .!!!")
    print(f"note: enter Prefect Workflow. {flow_trigger_dir=}, {config_json=}")

    flow_running_dirname = "free_energy_flow/"
    # flow_running_dir = os.path.join(flow_trigger_dir, flow_running_dirname)
    

    workflow_service_module = WorkflowServiceModule(
        flow_trigger_dir=flow_trigger_dir,
        flow_running_dirname=flow_running_dirname)

    my_io_handler = workflow_service_module.provide_file_handler()

    my_injector = Injector([configure_for_testing,
                            # DirnamesModule,
                            workflow_service_module
                            ])

    thermo_input:ThermoInputData = FreeEnergyLineWorkflowStart()
    print(f"FreeEnergyLineWorkflow: {thermo_input=}")
    # statistics_updates =  
    # updated_npt_input = thermo_input | {"nsteps": 100000, "stat_bsize": 100}

    print(f"note: thermo condition:{thermo_input=}")

    #note: pylance cannot recognize injectior
    with injection_context(my_injector):
        npt = NPTEquiSimulation( thermo_input._asdict() | {"nsteps": 40000, "stat_skip":500, "stat_bsize": 100})
        # npt_r = npt(skip_steps=['prepare', 'run']) # want to inject my_injector.get(WorkflowService)
        npt_r = npt()

        # npt.io_handler = my_io_handler

        # npt_r = npt()
        accurate_pv_value_from_npt = npt_r['pv']
        accurate_pv_err_value_from_npt = npt_r['pv_err']
        
        pv_dict = {
            'accurate_pv_value_from_npt':accurate_pv_value_from_npt,
            ' accurate_pv_err_value_from_npt': accurate_pv_err_value_from_npt
        }

        r1 = NPTResultToNVTConfLmp(header_print_num=100)()

        nvt = NVTEquiSimulation(updates=( thermo_input._asdict() | {"equi_conf":"npt_avg.lmp", "nsteps": 20000, "stat_bsize": 100}))
        nvt_r = nvt()
        # nvt_r = nvt(skip_steps=['prepare', 'run'])
        r2 = ExtractNVTToHTIConfLmp()()
        hti_sim = HTISimulation(thermo_input._asdict() 
                                | {"nsteps": 5000, "equi_conf": "out.lmp", "ref": "einstein", "switch":"three-step"}
                                | pv_dict)
        hti_r:HTIResultData = hti_sim()
        # hti_r:HTIResultData = hti_sim(skip_steps=['prepare', 'run'])

        print(f"hti_r {hti_r=} ")
        free_energy_value_point = hti_r['free_energy_value_point']

        ti_sim = TISimulation(thermo_input._asdict()
                          |{'nsteps': 30000}
                          |{'path': 't', 'temp_seq':["200:1800:100  ", "1800"]}
                          |{'free_energy_value_point': free_energy_value_point})
        
        r = ti_sim()

        # hti = HTISimulation(
        #     {'free_energy_value_point':free_energy_value_point,
        #      'manual_pv': pv}, )
        # hti_return = hti()
        # accurate_pv_from_npt = npt_return['pv']
        # hti_result_dict = hti.extract()
        # hti_to_ti_dict = HTIResultToTIDict(
        #     e1=hti_result_dict['e1'],
        #     e1_err=hti_result_dict['e1_error'],
        #     const_thermo_name='temp',
        #     # const_thermo_value= 
        # )
        # ti = TISimulation(hti_result_dict=hti_result_dict)
        # {}
        # {'Eo': }


        # r = ti(call_entity=thermo_input|{"dump_freq":10000, "nsteps":30000})
        # r = hti(call_entity=thermo_input | {"equi_conf": "out.lmp", "ref": "einstein", "switch":"three-step",})

    # 
    # print(f"npt.job_dir:{npt.job_dir}")

    # r2 = npt_converter.npt_result_to_nvt_conf_lmp(npt_dir=npt.job_dir)
    # print(f"r2:{r2}")

    # r3 = nvt(call_entity=thermo_input | {"equi_conf":"npt_avg.lmp", "nsteps": 30000, "stat_bsize": 100} )
    # print(f"r3:{r3}")

    # r4 = npt_converter.extract_nvt_to_hti_conf_lmp()

    # r3 = nvt(call_entity=)

    # with myinjector.injector():
    #     npt_converter = NptConverter()

    

    # r2 = injector.
    # r2 = injector.call_with_injection(npt_result_to_nvt_conf_lmp)

    # handler = injector.get(NptConverter)
    # r2 = handler.npt_result_to_nvt_conf_lmp(npt_dir=npt.job_dir)
    # r2 = npt_result_to_nvt_conf_lmp(npt_dir=npt.job_dir, io_handler=default_workflow_service_provider.io_handler)
    # r2 = npt_result_to_nvt_conf_lmp(npt_dir=npt.job_dir, io_handler=default_workflow_service_provider.io_handler)


    create_markdown_artifact(
        key="npt-report",
        markdown=markdown_report.format(r=json.dumps(r, indent=4)),
        description="Flow Run Report",
    )
    # nvt = NVTEquiSimulation()
    # r2 = nvt(r) 
    # r = npt.execute()
    # npt_r = npt()
    # nvt = NVTEquiSimulation()
    # nvt_r = nvt()
    # return nvt_r
    return r

default_flow_trigger_dir = os.path.join(os.path.dirname(__file__), "../../examples/" )

if __name__ == "__main__":
    # FreeEnergyLineWorkflow()
    FreeEnergyLineWorkflow.serve(name="dpti-workflow-line-deployment",
                      tags=["onboarding"],
                      parameters={
                                  "flow_trigger_dir": default_flow_trigger_dir,
                                  "config_json": "FreeEnergy.json", 
                                #   "refresh_cache": False
                                  },
                      pause_on_shutdown=False)
