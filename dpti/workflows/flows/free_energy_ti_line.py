#%%

from datetime import datetime
import random
from functools import cached_property, cache
from typing import TypeVar, Protocol, Generic, Any, Optional
from typing import List, Union, AbstractSet, MutableMapping
from typing import Annotated, Dict
from pathlib import Path
from pydantic import BaseModel, Field, AliasChoices, computed_field
from dataclasses import dataclass
from abc import ABC, abstractmethod
import yaml

from injector import Injector

# from dependency_injector import containers, providers
# from dependency_injector.wiring import Provide, inject
from prefect import flow
from typing import TypedDict
#%%
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, project_root)
print(f"project_root: {project_root=}")
# sys.path.insert(0, project_root)
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../' '../', '../')))
# print(f"sys.path: {sys.path=}")
import dpti
from dpti.workflows.service.di import injection_context
from dpti.workflows.flows.flow_settings import parse_yaml_like_input
# from dpti.workflows.simulations.base import SettingsBase
from dpti.workflows.service.workflow_decorator import workflow_task_decorator, DataProcessor
# from dpti.workflows.service.service_container import WorkflowContainer, BasicWorkflowServices
from dpti.workflows.service.workflow_service_module import WorkflowServiceModule, BasicWorkflowServices
from dpti.workflows.flows.base_flow import BaseWorkflow, FlowMetaInfo, FlowTriggerInfo, FlowRuntimeContext, FlowProcedureControl



#%%
from dpti.workflows.simulations.equi_sim import NPTEquiSimulationSettings, NVTEquiSimulationSettings, ExtractNVTForHTIConfLmp
from dpti.workflows.simulations.hti_sim import HTISimulationSettings
from dpti.workflows.simulations.ti_sim import TISimulationSettings
#%%
from dpti.workflows.simulations.equi_sim import NPTEquiSimulation, NVTEquiSimulation
from dpti.workflows.simulations.equi_sim import NPTResultToNVTConfLmp
from dpti.workflows.simulations.ti_sim import TISimulation
from dpti.workflows.simulations.hti_sim import HTISimulation


#%%


# 工作流设置类型，可根据需要扩展




class FreeEnergyFlowDomainInputRaw(BaseModel):
    config_yaml: str = Field(default="ti_line_beta144xy.yaml", description="some template yaml file")
    target_temp: Optional[float] = Field(default=None, description="None means just use the config file temperature")
    target_press: Optional[float] = Field(default=None, description="None means just use the config file pressure")
    ti_path: Optional[str] = Field(default=None, description="None means just use the config file ti_path")

class FreeEnergyFlowTriggerModel(BaseModel):
    flow_trigger_info: FlowTriggerInfo
    # workflow_services: WorkflowServices
    flow_procedure_control: FlowProcedureControl
    domain_input_raw: FreeEnergyFlowDomainInputRaw

class ThermoInputData(BaseModel, extra='ignore'):
    conf_lmp: str = Field(..., validation_alias=AliasChoices('conf_lmp', 'conf_file', 'equi_conf'))
    ti_path: str = Field(..., validation_alias=AliasChoices('ti_path', 'path'))
    target_pres: float
    target_temp: float
    ens: str
    model: str
    mass_map: list[float]
    if_liquid: Optional[bool] = Field(default=False)
    if_water: Optional[bool] = Field(default=False)
    if_meam: Optional[bool] = Field(default=False)
    meam_model: Optional[dict] = Field(default=None)
#%%



class FreeEnergyLineDomainInput(BaseModel):
    thermo_input:ThermoInputData
    npt_simulation_settings:NPTEquiSimulationSettings
    nvt_simulation_settings:NVTEquiSimulationSettings
    hti_simulation_settings:HTISimulationSettings
    ti_simulation_settings:TISimulationSettings

#%%




def load_default_yaml_like_asdict() -> Dict[str, Any]:
    yaml_like_file = os.path.join(os.path.dirname(__file__),
    '../examples/Sn_beta_quicktest/ti_line_beta144xy.yaml')
    default_yaml_like_asdict = parse_yaml_like_input(
        yaml_like_file, 
        FreeEnergyLineDomainInput
        ).model_dump()
    return default_yaml_like_asdict


print(f"loading default yaml for web browser:{load_default_yaml_like_asdict()=}")
# print(f"{load_default_yaml_like_asdict()=}")

class FreeEnergyFlowWorkorder(BaseModel): # implements WorkflowSettingsModel
    # flow_trigger_info: FlowTriggerInfo
    flow_run_number: int = Field(default=random.randint(114514, 1000000))
    flow_trigger_dir_raw: str = Field(default="../../workflows/examples/Sn_beta_quicktest/")
    flow_procedure_control: FlowProcedureControl = Field(default=FlowProcedureControl())
    # domain_input_raw: Field(default=FreeEnergyLineDomainInput)
    # domain_input_yaml_like: Union[Dict[str, Any], Path, str] = Field(default_factory=load_default_yaml_like)
    domain_input_yaml_like: Union[Dict[str, Any], Path, str] = Field(default=load_default_yaml_like_asdict())
    

    @computed_field
    @cached_property    
    def domain_input_raw(self) -> FreeEnergyLineDomainInput:  # Partial[FreeEnergyLineDomainInput]:
        domain_input_raw = parse_yaml_like_input(self.domain_input_yaml_like, FreeEnergyLineDomainInput)
        return domain_input_raw

    @computed_field
    @cached_property
    def flow_trigger_dir(self) -> str:
        flow_trigger_dir = os.path.abspath(self.flow_trigger_dir_raw)
        return flow_trigger_dir

    @computed_field
    @cached_property
    def flow_running_dirname(self) -> str:

        flow_run_number = self.flow_run_number
        ti_path = self.domain_input_raw.thermo_input.ti_path
        conf_name = self.domain_input_raw.thermo_input.conf_lmp.replace('.lmp', '')
        flow_running_dirname = (f"TI_{conf_name}"
                                f"_path_{ti_path}"
                                f"_temp{self.domain_input_raw.thermo_input.target_temp}K"
                                f"_pres{self.domain_input_raw.thermo_input.target_pres}bar"
                                f"_run{flow_run_number}/")
        return flow_running_dirname


class FlowSettingsBase(object):
    pass

class FreeEnergyLineSettings(FlowSettingsBase):
    # flow_run_info:FlowRunInfo
    # flow_meta_info:FlowMetaInfo
    # flow_trigger_info:FlowTriggerInfo 

    flow_trigger_dir: str
    domain_input_raw: FreeEnergyLineDomainInput 
    flow_runtime_context: Optional[FlowRuntimeContext]

    def __init__(self,
        flow_trigger_dir: str,
        flow_running_dirname: str,
        domain_input_raw: FreeEnergyLineDomainInput, # unvalid partial
        flow_runtime_context: Optional[FlowRuntimeContext] ,
    ):
        self.flow_trigger_dir = flow_trigger_dir
        self.domain_input_raw = domain_input_raw
        self.flow_runtime_context = flow_runtime_context

        # self.flow_running_dirname = self.flow_trigger_info.flow_running_dirname
        base_thermo_updates = self.base_thermo_updates
        self.flow_running_dirname = flow_running_dirname


    @cached_property
    def flow_running_dir(self) -> str:
        flow_running_dir = os.path.join(self.flow_trigger_dir, self.flow_running_dirname)
        return flow_running_dir

    @cached_property
    def base_thermo_updates(self) -> Dict[str, Any]:
        t = self.domain_input_raw.thermo_input
        base_thermo_updates = {
            'conf_lmp':t.conf_lmp,
            'temp':t.target_temp,
            'pres':t.target_pres,
            'model':t.model,
            'mass_map':t.mass_map,
            'if_liquid':t.if_liquid,
            'if_water':t.if_water,
            'if_meam':t.if_meam,
            'meam_model':t.meam_model,
        }
        return base_thermo_updates

    @cached_property
    def npt_simulation_settings(self):
        npt_simulation_settings_raw = self.domain_input_raw.npt_simulation_settings
    
        npt_simulation_settings = NPTEquiSimulationSettings(
            **(npt_simulation_settings_raw.model_dump()
               | self.base_thermo_updates
               | {'equi_conf': self.domain_input_raw.thermo_input.conf_lmp}
               | {'ens':self.domain_input_raw.thermo_input.ens})
        )
        return npt_simulation_settings


    @cached_property
    def nvt_simulation_settings(self):
        nvt_simulation_settings = NVTEquiSimulationSettings(
            **(self.domain_input_raw.nvt_simulation_settings.model_dump()
               | self.base_thermo_updates
               | {'equi_conf': 'npt_avg.lmp'} # always use npt_avg.lmp for nvt simulation
        ))
        return nvt_simulation_settings

    @cached_property
    def hti_simulation_settings(self):
        # note NVT simulation decide the dump file and HTI init  lmp file
        nvt_if_dump_avg_posi = self.domain_input_raw.nvt_simulation_settings.if_dump_avg_posi
        if nvt_if_dump_avg_posi:
            equi_conf = "nvt_last_dump_avgposi.lmp"
        else:
            equi_conf = "nvt_last_dump.lmp"

        hti_simulation_settings = HTISimulationSettings(
            **(self.domain_input_raw.hti_simulation_settings.model_dump()
               | self.base_thermo_updates
               | {'equi_conf': equi_conf}
               | {'ens':self.domain_input_raw.thermo_input.ens})
        )
        return hti_simulation_settings

    @cached_property
    def ti_simulation_settings(self):
        ti_simulation_settings = TISimulationSettings(
            **(self.domain_input_raw.ti_simulation_settings.model_dump()
               | self.base_thermo_updates
               | {'path': self.domain_input_raw.thermo_input.ti_path, }
               | {'equi_conf': self.domain_input_raw.thermo_input.conf_lmp}
               | {'ens':self.domain_input_raw.thermo_input.ens})
        )
        return ti_simulation_settings


#%%




#%%




class FreeEnergyWorkflow(BaseWorkflow):
    flow_settings: FreeEnergyLineSettings
    flow_runtime_context: FlowRuntimeContext


    def __init__(self, flow_workorder: FreeEnergyFlowWorkorder):
    # def init_settings_from_workorder(self, flow_workorder: FreeEnergyFlowWorkorder):
        self.flow_workorder = flow_workorder

        flow_running_dir = os.path.join(flow_workorder.flow_trigger_dir, flow_workorder.flow_running_dirname)

        self.flow_runtime_context = FlowRuntimeContext(
            flow_running_dir=flow_running_dir,
            flow_meta_info=FlowMetaInfo(
                flow_platform= "prefect",
                flow_name=self.__class__.__name__,
                flow_version="0.1.0"
            ),
            flow_procedure_control=flow_workorder.flow_procedure_control,
            flow_workorder=flow_workorder,
        )

        self.flow_trigger_dir = flow_workorder.flow_trigger_dir
        self.flow_running_dirname = flow_workorder.flow_running_dirname
        
        free_energy_line_settings = FreeEnergyLineSettings(
            flow_trigger_dir=self.flow_trigger_dir,
            flow_running_dirname=self.flow_running_dirname,
            domain_input_raw=flow_workorder.domain_input_raw,
            flow_runtime_context=self.flow_runtime_context,
        )

        self.flow_settings = free_energy_line_settings

    # @inject
    # def init_workflow_services(self,
    #         workflow_services:BasicWorkflowServices,
    #     ):
    #     self.workflow_services = workflow_services

    def init_workflow_services_injector(self, workflow_services_injector: Injector):
        self.workflow_services_injector = workflow_services_injector
        self.workflow_services = workflow_services_injector.get(BasicWorkflowServices)
        self.workflow_services_injector.binder.bind(FlowRuntimeContext, to=self.flow_runtime_context)
        # self.workflow_services_injector.binder.bind('flow_running_dir', to=self.flow_trigger_dir)
        # self.workflow_services_injector.binder.bind('flow_runtime_context', to=self.flow_runtime_context)

    def init_workflow_context(self):
        """初始化工作流上下文"""
        workflow_context = {}

    def for_json(self):
        return {
            "flow_workorder": self.flow_workorder.model_dump(),
            # "flow_settings": self.flow_settings.for_json(),
            "flow_runtime_context": self.flow_runtime_context,
            # "flow_procedure_control": self.flow_procedure_control.model_dump(),
            "domain_input_raw": self.domain_input_raw.model_dump(),
        }


    def init_ops_input_params(self):
        pass

    def _execute_impl(self) -> Any:
        # 使用注入的服务
        injector = self.workflow_services_injector
        with injection_context(injector):
            result = self.flow_settings.domain_input_raw

            return_flow_start_check = self.flow_start_check() # type: ignore

            npt_equi_simulation = NPTEquiSimulation(
                node_settings=self.flow_settings.npt_simulation_settings)

            r = npt_equi_simulation()
            npt_avg_lmp = NPTResultToNVTConfLmp(header_print_num=100)()
            nvt = NVTEquiSimulation(
                node_settings=self.flow_settings.nvt_simulation_settings)

            nvt_r = nvt(prev_results={'avg_conf_from_npt':npt_avg_lmp})

            if_dump_avg_posi = self.flow_settings.domain_input_raw.nvt_simulation_settings.if_dump_avg_posi

            nvt_extracted_lmp = ExtractNVTForHTIConfLmp(if_dump_avg_posi=if_dump_avg_posi)()

            hti_sim = HTISimulation(
                node_settings=self.flow_settings.hti_simulation_settings)

            hti_r = hti_sim(prev_results={'conf_file':nvt_extracted_lmp})

            ti_sim = TISimulation(
                node_settings=self.flow_settings.ti_simulation_settings)

            ti_r = ti_sim(prev_results={'free_energy_ref_value_point':hti_r['free_energy_ref_value_point']})

        print(f"{result=}")
        print(f"{nvt_r=}")
        return result
#%%
class Configuration:
    def __init__(self, file_handler_string):
        self.file_handler_string = file_handler_string

local_file_config = Configuration(':localfile:')
def configure_for_localtesting(binder):
    configuration = local_file_config
    # binder.bind(Configuration, to=configuration, scope=singleton)
    binder.bind(Configuration, to=configuration)
    # binder.bind

#%%
@flow(log_prints=True,
    persist_result=True,
    # flow_run_name=lambda: (
    #     f"free_energy_workflow-{datetime.now().strftime('%Y%m%d_%H%M%S')}-r{format(random.getrandbits(24), '06X')}"
    #             ),)
    flow_run_name="free_energy_workflow-{flow_workorder.flow_running_dirname}"
    )
def free_energy_workflow_flow(flow_workorder: FreeEnergyFlowWorkorder):
    # container = WorkflowContainer()
    # container.wire(modules=[__name__,
    # "dpti.workflows.simulations.equi_sim"])
    flow_running_dirname = flow_workorder.flow_running_dirname
    workflow_service_module = WorkflowServiceModule(
        flow_trigger_dir=flow_workorder.flow_trigger_dir,
        flow_running_dirname=flow_running_dirname,
    )

    my_injector = Injector([configure_for_localtesting, workflow_service_module])



    # basic_workflow_services = workflow_service_module.provide_basic_workflow_services()
    # basic_workflow_services = my_injector.get(BasicWorkflowServices)

    free_energy_workflow = FreeEnergyWorkflow(flow_workorder=flow_workorder)
    # free_energy_workflow.init_workflow_services(workflow_services=basic_workflow_services)
    free_energy_workflow.init_workflow_services_injector(workflow_services_injector=my_injector)
    # free_energy_workflow.init_settings_from_workorder(flow_workorder=flow_workorder)
    free_energy_workflow._execute_impl()


# 使用示例
if __name__ == "__main__":
    # 创建工作流实例
    
    # 直接执行
    # workflow.flow_object()
    # 部署

    # workflow.
    free_energy_workflow_flow.serve(
        name="dpti-test3-workflow", 
        # parameters={
        #     "config_yaml": "FreeEnergy.yaml",
        #     "flow_trigger_dir": "./examples",
        #     "flow_run_number": 0
        # }
        )
# %%
#%%
    
