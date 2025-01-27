#%%
import glob
import json
from lib2to3.fixes.fix_tuple_params import tuple_name
import os
import sys
# from airflow.models import DAG
from datetime import datetime
# from tkinter import NO
from textwrap import indent
from tkinter import NO
import typing

from typing import ClassVar, Literal, Dict, List, Any, Protocol, Tuple, TypeVar, Generic, Union, Iterator
from typing import Optional, get_args, overload, NoReturn, Callable, NewType, Annotated
from typing import NamedTuple, Tuple
from unittest import skip
from attr import dataclass
# from flask.scaffold import F
import injector
import numpy as np
# from regex import D
from typing_extensions import Type, TypedDict
from collections import defaultdict

# from altair import Type
# from airflow.decorators import dag, task
# from airflow.operators.python import get_current_context

# from dpdispatcher.lazy_local_context import LazyLocalContext
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../', '../')))
print(sys.path)
from dpti import equi, hti, hti_liq, ti

from prefect.artifacts import create_link_artifact, create_markdown_artifact
from pydantic import AliasChoices, BaseModel, Field, ValidationError
import weakref
import functools
# from pathlib import Path
from dpti.lib.utils import create_path, parse_seq
from abc import ABC, abstractmethod
# from pydantic_partial import create_partial_model
# from fs.osfs import OSFS
# from fs import open_fs
from contextlib import contextmanager
from injector import provider, Injector, inject, singleton
from injector import Module
# from pydantic_partial import PartialModelMixin

from functools import partial, singledispatchmethod, wraps


from prefect import flow, task
from prefect import Task
# from prefect.tasks import task_input_hash
# from dpti.workflows.prefect.prefect_task_hash import task_input_json_hash
# from dpti.workflows import workflow_service


REFRESH_CACHE = True
# REFRESH_CACHE = False

#%%
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
print(f"{PROJECT_ROOT=}")
sys.path.insert(0, PROJECT_ROOT)
# from dpti.workflows.job_executor import JobExecutor, DpdispatcherExecutor
from dpti.workflows.service.job_executor import JobExecutor, DpdispatcherExecutor
from dpti.workflows.service.file_handler import IOHandler, LocalFileHandler, FilesToUploadEntity
from dpti.workflows.service.di import InjectionContext, context_inject, injection_context
from dpti.workflows.service.workflow_service_module import WorkflowServiceModule, WorkflowService

from dpti.workflows.simulations.equi_sim import NPTEquiSimulation, NVTEquiSimulation
from dpti.workflows.simulations.hti_sim import HTISimulation, HTIResultData
from dpti.workflows.simulations.ti_sim import TISimulation
from dpti.workflows.prefect_task_hash import task_input_json_hash

#%%
# json.dumps



#%%
# import logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# from pydantic.main import Partial
# from injector import Injector
# used by localhost
# prefect 
# PREFECT_API_URL="http://127.0.0.1:4200/api"

# from dependency_injector import containers, providers
# from dependency_injector.wiring import inject, Provide

#%%


# class BaseInput(BaseModel):
#     def validate_not_none(self):
#         missing_fields = [field for field, value in self.__dict__.items() if value is None]
#         if missing_fields:
#             raise ValueError(f"The following fields are not set: {', '.join(missing_fields)}")
#         else:
#             pass

#%%





#%%

# class ModelDumpProtocol(Protocol):
#     def model_dump(self) -> Dict[str, Any]: ...

# class PydanticStyleTypedDict(TypedDict):
#     pass

# def model_dump(obj: PydanticStyleTypedDict) -> Dict[str, Any]:
#     return dict(obj)


# PydanticStyleTypedDictWithDump = PydanticStyleTypedDict & ModelDumpProtocol

#%%


# @dataclass

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

EnsembleType = Literal['npt-xy', 'nvt', 'npt-iso']
ensemble_field = Field(..., description="Must be one of 'npt-xy', 'nvt', 'npt-iso'")

# class ThermoConditionInitEntity(BaseModel, extra='allow'):
#     # flow_running_dir: str
#     equi_conf: str
#     temp: float
#     pres: float
#     ens: EnsembleType = ensemble_field

class ThermoConditionCallEntity(BaseModel):
    pass

# class ThermoConditionReturn(BaseModel):
#     pass

#%%


# DpdispatcherExecutor
#%%
    
# class Parent():
#     @classmethod
#     def subclass(cls, classname):
#         subclass_map = {subclass.__name__: subclass for subclass in cls.__subclasses__()}
#         subclass = subclass_map[classname]
#         # instance = super(Parent, subclass).__new__(subclass)
#         return subclass

# class Child1(Parent):
#     feature = 1
#     def __init__(self, feat=1):
#         self.feat = feat
#         pass


# class Child2(Parent):
#     feature = 2

# a = Parent.subclass("Child1")(feat=3)  # <class '__main__.Child2'>
# # r = Child1(feat='tt')


#%%




# class CreateFromTemplateMixinMeta(type):
#     def __call__(cls, **kwargs) -> Type:
#         # 直接修改传入的 cls
#         cls.TEMPLATE_DEFAULT_JSON = kwargs['TEMPLATE_DEFAULT_JSON']
#         cls.TEMPLATE_ADDITIONAL_REQUIRED_FIELDS = kwargs['TEMPLATE_ADDITIONAL_REQUIRED_FIELDS']

#         # 创建一个新的类，继承自修改后的 cls
#         configured_mixin_cls:Type = type(f"Configured{cls.__name__}", (cls,), {})
#         return configured_mixin_cls

# class CreateFromTemplateMixinMeta(type):
#     def __new__(mcs, name, bases, attrs, **kwargs):
#         # template_mixin = next((b for b in bases if isinstance(b, D)), None)
#         TEMPLATE_DEFAULT_JSON = kwargs['TEMPLATE_DEFAULT_JSON']
#         TEMPLATE_ADDITIONAL_REQUIRED_FIELDS = kwargs['TEMPLATE_ADDITIONAL_REQUIRED_FIELDS']
#         # if template_mixin:
#         attrs['TEMPLATE_DEFAULT_JSON'] = TEMPLATE_DEFAULT_JSON
#         attrs['TEMPLATE_ADDITIONAL_REQUIRED_FIELDS'] = TEMPLATE_ADDITIONAL_REQUIRED_FIELDS
#         return super().__new__(mcs, name, bases, attrs)


        # if 'CreateFromTemplateMixin' not in kwargs:
        #     raise TypeError(f"{name}: CreateFromTemplateMixin configuration must be specified")
        
        # config = kwargs['CreateFromTemplateMixin']
        # if 'DEFAULT_TEMPLATE_JSON' not in config:
        #     raise TypeError(f"{name}: DEFAULT_TEMPLATE_JSON must be specified in CreateFromTemplateMixin configuration")
        # if 'TEMPLATE_ADDITIONAL_REQUIRED_FIELDS' not in config:
        #     raise TypeError(f"{name}: TEMPLATE_ADDITIONAL_REQUIRED_FIELDS must be specified in CreateFromTemplateMixin configuration")
        
        # attrs['DEFAULT_TEMPLATE_JSON'] = config['DEFAULT_TEMPLATE_JSON']
        # attrs['TEMPLATE_ADDITIONAL_REQUIRED_FIELDS'] = config['TEMPLATE_ADDITIONAL_REQUIRED_FIELDS']
        
        # return super().__new__(mcs, name, bases, attrs)







#%%


class FreeEnergyLineWorkflowInput(BaseModel):
    target_temp: float
    target_pres: float
    work_base_dir: str
    path: str
    conf_lmp: str
    ens: str
    if_liquid: bool
    npt_conf: str

# npt_equi_input_data = NPTEquiSimulationData.from_template(updates={})

# EquiLammpsSettings = type('EquiLammpsSettings', (EquiLammpsInput, EquiLammpsAnalyze), {})



#%%
# DEFAULT_EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), '../examples/')
# class B:
#     pass

# class A(object):
#     def for_json(self):
#         d = {'a':3, 'b':4}
#         return d
#     pass

# a = A()

# json.dumps(a, for_json=True)

# a = A()

# print(a.__class__)

# print(a.__class__.__base__)
# t = list(a.__class__.__mro__)
# print(t)
# t.append(BaseModel)
# # a.__class__.__mro__=tuple(t)
# print(a.__class__.__mro__)

# print(isinstance(a, BaseModel))
# print(isinstance(NPTEquiSimulation(), NPTEquiSimulation))
# print(isinstance(NPTEquiSimulation(), int))
# print(isinstance(NPTEquiSimulation(), BaseModel))
#%%



#%%

#%%




#%%


# def __init__(self, key):
#         self.key = key
#         MyClass._instances[key] = self  # 在初始化时将实例添加到字典中

#     @classmethod
#     def get_instance(cls, key):
#         return cls._instances.get(key) 
# class 

#%%

# class HTIResultToTIDict(TypedDict):
# class HTIResultToTIData(NamedTuple):
#     pass
#     e1: float # Helmholtz(Gibbs) free energy[in eV]. 
#     e1_err: float # standard deviation of e1
#     temp: float # the e1 corresponding thermo condition
#     pres: float # the e1 corresponding thermo condition


    # const_thermo_name: str # along pressure: `temp` `, along temperature: `pres` 
    # const_thermo_value: float # along pressure: temperature value, along pressure: pressure value, 

    # @classmethod
    # def convert_from_hti_calculation_dict(cls, hti_result_dict:HTIResultDict, thermo_input):
    #     instance = cls(
    #         e1=hti_result_dict['e1'],
    #         e1_err=hti_result_dict['e1_err'],
    #         temp=thermo_input['temp'],
    #         pres=thermo_input['pres'],
    #         )
    #     return instance
    



# ti.post_tasks()

# class HTIresultToTi

# def extract_hti_result_to_ti



# class 

# class 

#%%







    
    # sef
    # def parse(self):
    #     pass
        # parse_seq()
        
    pass

# self.seq_dict.add_integration_path()

# path_deep_on = HTIIntegraionPath(field_name='lambda_deep_on', )





#     'field_name': 'lambda_lj_on',
    
# }


# class StepStrategy(ABC):
#     integration_path_list = []
#     @abstractmethod
#     def execute(self, d) -> Any:
#         pass

# class OneStepStrategy(StepStrategy):
#     def execute(self, d):
#         return {'lambda_deep_on': parse_seq(d.lambda_deep_on)}

# class TwoStepStrategy(StepStrategy):
#     def __init__(self, updated_nodedata):
#         self.d = updated_nodedata
#         pass
#     def execute(self, d):
#         self.integration_path_list.append(
#             path_deep_on.parse_seq_list(seq_list=d.lambda_deep_on))

#         return {
#             'lambda_deep_on': parse_seq(d.lambda_deep_on),
#             'lambda_spring_off': parse_seq(d.lambda_spring_off)
#         }

# class ThreeStepStrategy(StepStrategy):
#     def execute(self, d):
#         return {
#             'lambda_deep_on': parse_seq(d.lambda_deep_on),
#             'lambda_spring_off': parse_seq(d.lambda_spring_off),
#             'lambda_lj_on': parse_seq(d.lambda_lj_on)
#         }


# class HTIIntegrations(object):
#     def __init__(self, switch):
#         self.switch = switch
#         self.seq_dict = {}
#         self.integration_path_list = []
#         # self.
#     def use_lambda(self):
#         pass

#     def choose_to_execute(self, instance: HTISimulation):

#         d = instance.updated_nodedata
#         match self.switch:
#             case 'one-step':
#                 self.integration_path_list = [path_deep_on]
#                 # self.integration_path_list = [path_lj_on, path_deep_on, path_spring_off]
#                 # self.seq_dict['lambda_deep_on'] = parse_seq(d.lambda_deep_on)
#             case 'two-step':
#                 self.integration_path_list = [path_deep_on, path_spring_off]
#                 # self.seq_dict['lambda_deep_on'] = parse_seq(d.lambda_deep_on)
#                 # self.seq_dict['lambda_spring_off'] = parse_seq(d.lambda_spring_off)
#             case 'three-step':
#                 self.integration_path_list = [path_lj_on, path_deep_on, path_spring_off]
#                 # self.seq_dict['lambda_deep_on'] = parse_seq(d.lambda_deep_on)
#                 # self.seq_dict['lambda_spring_off'] = parse_seq(d.lambda_spring_off)
#                 # self.seq_dict['lambda_lj_on'] = parse_seq(d.lambda_lj_on)
#             case _:
#                 raise ValueError(f"Error option {self.switch=}")
#         return self.integration_path_list
    
#     def add_seq_list(self, path_config_dict) -> None:
#         for integration_path in self.integration_path_list:
#             field_name = integration_path.field_name
#             integration_path.parse_seq_list(path_config_dict[field_name])

#     def generate_subtasks(self, io_handler):
#         pass
    



    # def 

#%%

            #     all_lambda = parse_seq(jdata["lambda_deep_on"])
            # elif step == "spring_off":
            #     all_lambda = parse_seq(jdata["lambda_spring_off"])
            # elif step == "lj_on":
            #     all_lambda = parse_seq(jdata["lambda_lj_on"])

    

#%%
# # 
# a = dict(d=3,c=2)

# markdown_report = f"This flow return `info`:  {a}"
# print(markdown_report)


# print(deserialized_data)


markdown_report = """This flow return `info`:  {r}
![Logo Image](https://github.com/deepmodeling/deepmd-kit/raw/r2/doc/_static/logo.svg)
result: /home/felix/1_software/dpti/examples/NPT_sim/new_job/result
"""


#%%



# class MyDecorator:
#     def __init__(self, func):
#         self.func = func

#     def __call__(self, *args, **kwargs):
#         print("Something is happening before the method is called.")
#         result = self.func(*args, **kwargs)
#         print("Something is happening after the method is called.")
#         return result

#%%


# class InjectableFunction




    # def __call__(self) -> Any : pass
        

# class NptConverter(metaclass=InjectableMeta):
#     @overload
#     def __init__(self) -> NoReturn: ...

#     @overload
#     def __init__(self, io_handler: IOHandler):...

#     @inject
#     def __init__(self, io_handler: IOHandler | None = None ):
#                 #  npt_dir: Annotated[str, 'npt_dir'],
#                 #  nvt_dir: Annotated[str, 'nvt_dir']
#         if io_handler is None:
#             raise ValueError("io_handler must be provided and cannot be None."
#                              + "Possible due to Dependency Injection failed"
#                              + "")
#         self.io_handler = io_handler
        
    
    
        # self.npt_dir = npt_dir
        # self.nvt_dir = nvt_dir

    # @inject
    # def npt_result_to_nvt_conf_lmp(self) -> Any:
        
    #     self.io_handler.use_job_info(
    #         job_dirname=NVTEquiSimulation.JOB_DIRNAME)
        
    #     npt_avg_conf_lmp = equi.npt_equi_conf(
    #         npt_dir=os.path.join(self.io_handler.flow_running_dir, 
    #                              NPTEquiSimulation.JOB_DIRNAME))
    #     r_lmp = self.io_handler.write_pure_file(file_path="npt_avg.lmp",
    #                                             file_content=npt_avg_conf_lmp)
    #     print(f"header for: npt_avg.lmp:{npt_avg_conf_lmp[0:100]}")
    #     return r_lmp

    # @inject
    # def extract_nvt_to_hti_conf_lmp(self) -> Any:
    #     self.io_handler.use_job_info(job_dirname="HTI_sim/new_job/")
    #     self.io_handler.upload_files(file_paths = [NVTEquiSimulation.JOB_DIRNAME + "/out.lmp"],
    #                                  base_dir=self.io_handler.flow_running_dir)

# def injectable_function(cls):
#     @wraps(cls)
#     def wrapper(io_handler: IOHandler):
#         instance = cls(io_handler=io_handler)
#         return instance()
#     return wrapper

# @injectable_function
# class npt_result_to_nvt_conf_lmp(BaseFuncInjectable):
# class NPTResultToNVTConfLmp(BaseFuncInjectable):

class NPTResultToNVTConfLmp(object):
    def __init__(self, header_print_num: int = 100):
        self.header_print_num = header_print_num
        # self.io_handler.use_job_info(job_dirname=NVTEquiSimulation.JOB_DIRNAME)
        # npt_avg_conf_lmp = equi.npt_equi_conf(
        #     npt_dir=os.path.join(self.io_handler.flow_running_dir,
        #                          NPTEquiSimulation.JOB_DIRNAME))
        # print(f"header for: npt_avg.lmp:{npt_avg_conf_lmp[0:header_print_num]}")
        # return r_lmp
    
    @context_inject
    def __call__(self, io_handler:IOHandler) -> str:
        self.io_handler = io_handler
        self.io_handler.use_job_info(job_dirname=NVTEquiSimulation.JOB_DIRNAME)
        npt_avg_conf_lmp = equi.npt_equi_conf(
            npt_dir=os.path.join(self.io_handler.flow_running_dir,
                                 NPTEquiSimulation.JOB_DIRNAME))
        r_lmp = self.io_handler.write_pure_file(file_path="npt_avg.lmp", file_content=npt_avg_conf_lmp)
        return r_lmp


# class extract_nvt_to_hti_conf_lmp(BaseFuncInjectable):
# class ExtractNVTToHTIConfLmp(BaseFuncInjectable):
class ExtractNVTToHTIConfLmp(object):
    def __init__(self):
        pass

    @context_inject
    def __call__(self, io_handler:IOHandler) -> str:
        self.io_handler = io_handler
        self.io_handler.use_job_info(job_dirname="HTI_sim/new_job/")
        conf_lmp = os.path.join(NVTEquiSimulation.JOB_DIRNAME, "out.lmp")
        r = self.io_handler.upload_files(file_paths = [conf_lmp],
                                     base_dir=self.io_handler.flow_running_dir)
        return r[0]
    



# @inject
# def npt_result_to_nvt_conf_lmp(io_handler: IOHandler) -> str:
#     io_handler.use_job_info(job_dirname="NVT_sim/new_job/")
#     npt_avg_conf_lmp = equi.npt_equi_conf(
#         npt_dir=os.path.join(io_handler.job_dir,
#                              NPTEquiSimulation.JOB_DIRNAME))
#     r_lmp = io_handler.write_pure_file(file_path="npt_avg.lmp", file_content=npt_avg_conf_lmp)
#     print(f"header for: npt_avg.lmp:{npt_avg_conf_lmp[0:100]}")
#     return r_lmp

# def injectable(func):
#     class Wrapper(metaclass=InjectableMeta):
#         @inject
#         def __init__(self, **kwargs):
#             self.__dict__.update(kwargs)
        
#         def __call__(self, *args, **kwargs):
#             return func(self, *args, **kwargs)
    
#     return Wrapper

# @injectable
# def npt_result_to_nvt_conf_lmp(self, io_handler: IOHandler, npt_dir: str):
#     io_handler.use_job_info(job_dirname="NVT_sim/new_job/")
#     npt_avg_conf_lmp = equi.npt_equi_conf(
#         npt_dir=os.path.join(self.io_handler.job_dir, npt_dir))
#     r_lmp = self.io_handler.write_pure_file(file_path="npt_avg.lmp", file_content=npt_avg_conf_lmp)
#     print(f"header for: npt_avg.lmp:{npt_avg_conf_lmp[0:100]}")
#     return r_lmp

# class npt_result_to_nvt_conf_lmp(metaclass=InjectableMeta):
#     @inject
#     def __init__(self, io_handler: IOHandler, npt_dir: str):
#         self.io_handler = io_handler
#         self.npt_dir = npt_dir
#     def __call__(self):
#         self.io_handler.use_job_info(job_dirname="NVT_sim/new_job/")
#         npt_avg_conf_lmp = equi.npt_equi_conf(
#             npt_dir=os.path.join(self.io_handler.job_dir, self.npt_dir))
#         r_lmp = self.io_handler.write_pure_file(file_path="npt_avg.lmp", file_content=npt_avg_conf_lmp)
#         print(f"header for: npt_avg.lmp:{npt_avg_conf_lmp[0:100]}")
#         return r_lmp



#%%


class BaseBridge():
    pass



#%%




class Configuration:
    def __init__(self, file_handler_string):
        self.file_handler_string = file_handler_string

local_file_config = Configuration(':localfile:')

def configure_for_testing(binder):
    configuration = local_file_config
    # binder.bind(Configuration, to=configuration, scope=singleton)
    binder.bind(Configuration, to=configuration)

# def configure_for_dirs(binder):
#     binder.bind(Annotated[str, 'npt_dir'], to=NPTEquiSimulation.JOB_DIRNAME)
#     binder.bind(Annotated[str, 'nvt_dir'], to=NVTEquiSimulation.JOB_DIRNAME)




#%%
# my_injector = Injector([DirnamesModule])
                            


class FreeEnergyFlow(BaseModel):
    conf_lmp: str
    target_temp: int
    target_pres: int
    work_base_dir: str
    ti_path: str
    ens: str
    if_liquid: bool


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


# class FreeEnergyLine(BaseModel):
#     id: int
#     name: str
#     signup_ts: str = None
#     friends: list[int] = []

#%%

# npt = NPTEquiSimulation()

# #%%

# # %%

# print(type(npt))
# print(npt)
# print(dir(npt))

# #%%


# print(type(npt.execute))
# print(npt.execute)
# print(dir(npt.execute))

# %%
