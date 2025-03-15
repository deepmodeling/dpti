#%%
import os
import json
from typing import Dict, Any, List, Optional, Set
import yaml
from dataclasses import dataclass
from abc import ABC, abstractmethod
from prefect import flow, get_run_logger
from prefect.artifacts import create_markdown_artifact
from pydantic import BaseModel, Field, computed_field
from functools import cached_property
from typing import TypeVar, Protocol, Generic, Any, Optional, List, Union, AbstractSet, MutableMapping

# from dependency_injector import containers, providers
# from dependency_injector.wiring import Provide, inject
import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../' '../', '../')))
# from dpti.workflows.service.service_container import WorkflowContainer, WorkflowServices
from dpti.workflows.service.workflow_decorator import workflow_task_decorator
from dpti.workflows.service.workflow_service_module import WorkflowServiceModule
from dpti.workflows.service.workflow_service_module import WorkflowServices, BasicWorkflowServices

from dpti.workflows.simulations.base import SettingsBase, FlowRuntimeContext, FlowMetaInfo, FlowProcedureControl


#%%

# WorkflowSettingsType = TypeVar('WorkflowSettingsType')

# class FlowTriggerModel(BaseModel):
#     config_yaml: str
#     flow_trigger_dir: str
#     flow_run_number: int



# class WorkflowSettings(BaseModel):
#     config_yaml: str
#     flow_trigger_dir: str
#     flow_run_number: int

# class FlowRunInfo(BaseModel):
#     flow_run_number:Optional[int] = Field(default=0)
#     flow_trigger_dir:Optional[str] = Field(default=None)
#     flow_running_dirname:Optional[str] = Field(default=None)


# class FlowProcedureControl(BaseModel):
#     skip_steps: List[str] = Field(default=[])
#     dry_run: bool = Field(default=False)
#     debug_mode: bool = Field(default=False)

#     @computed_field
#     @cached_property
#     def skip_steps(self) -> List[str]:
#         if self.only_extract:
#             skip_steps = ['prepare', 'run']
#         else:
#             skip_steps = []
#         return skip_steps


@dataclass
class FlowTriggerInfo:
    flow_run_number: int
    flow_trigger_dir_raw: str
    flow_trigger_dir: str



class workflow_context(BaseModel):
    flow_meta_info: FlowMetaInfo
    flow_trigger_info: FlowTriggerInfo
    # workflow_services: WorkflowServices  # Protocol 类型
    flow_procedure_control: FlowProcedureControl
    
    # model_config = {
    #     "arbitrary_types_allowed": True  # 允许任意类型
    # }
    
# flow_params



DomainInputType = TypeVar('DomainInputType', bound=BaseModel)



class WorkflowSettingsModel(Protocol):
    """需要满足的最小接口要求"""
    flow_trigger_info: FlowTriggerInfo
    flow_procedure_control: FlowProcedureControl
    domain_input_raw: BaseModel



class BaseWorkflow(Generic[DomainInputType]):
    workflow_services: WorkflowServices
    flow_trigger_dir: str
    domain_input_raw: DomainInputType

    flow_runtime_context: FlowRuntimeContext

    flow_run_number: int = 0
    # @inject
    # def __init__(self, workflow_services:WorkflowServices=Provide[WorkflowContainer.workflow_services]):
        # """初始化工作流应用flow装饰器"""
        # 应用flow装饰器
        # self.workflow_services = workflow_services
        
    # def init_workflow_services(self, workflow_services:WorkflowServices):
    #     self.workflow_services = workflow_services

        # self.flow_object = flow(
        #     name=f"{self.__class__.__name__}Flow",
        #     # flow_run_name=,
        #     log_prints=True,
        #     persist_result=True
        # )(self.execute_flow)




    # def __call__(self, flow_params: WorkflowSettingsModel) -> Any:
    #     """执行工作流"""
    #     r = self.execute_flow(flow_params=flow_params)
    #     return r


    @workflow_task_decorator
    def flow_start_check(self):
        flow_trigger_dir = self.flow_trigger_dir
        flow_run_number = self.flow_run_number

        # config_yaml = self.flow_params.domain_params.config_yaml
        
        io_handler = self.workflow_services.io_handler

        self.check_if_flow_trigger_dir_exists()
        produced_flow_running_dir = self.workflow_services.io_handler.setup_flow_running_dir()
        # io_handler.create_job_dir()

        return produced_flow_running_dir

    def check_if_flow_trigger_dir_exists(self):
        flow_trigger_dir = self.flow_trigger_dir
        io_handler = self.workflow_services.io_handler
        if not io_handler.isdir(flow_trigger_dir):
            raise ValueError(f"{flow_trigger_dir=} is not a directory")
        return True

    # def check_if_config_yaml_exists(self, io_handler: IOHandler):
    #     pass
        
    def execute_flow(self) -> Any:
        """执行工作流"""
        r = self._execute_impl()
        return r
    
    @abstractmethod
    def _execute_impl(self) -> Any:
        """执行工作流"""
        pass




    # @abstractmethod
    # def _execute_impl(self, domain_input_raw: DomainInputRawType) -> Any:
    #     """执行工作流"""
    #     pass

#%%
