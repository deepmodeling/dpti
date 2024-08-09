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
from click import Option
from flask.scaffold import F
import injector
import numpy as np
from regex import D
from typing_extensions import Type, TypedDict
from collections import defaultdict

# from altair import Type
# from airflow.decorators import dag, task
# from airflow.operators.python import get_current_context

# from dpdispatcher.lazy_local_context import LazyLocalContext
from dpti import equi, hti, hti_liq, ti

from prefect.artifacts import create_link_artifact, create_markdown_artifact
from pydantic import AliasChoices, BaseModel, Field, ValidationError
import weakref
import functools
# from pathlib import Path
from dpti.lib.utils import create_path, parse_seq
from abc import ABC, abstractmethod
from pydantic_partial import create_partial_model
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
from prefect_task_hash import task_input_json_hash
DEFAULT_EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), '../../examples/')
REFRESH_CACHE = False

#%%

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
# from dpti.workflows.job_executor import JobExecutor, DpdispatcherExecutor
from job_executor import JobExecutor, DpdispatcherExecutor

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



def create_relative_symlink_file(file_path, target_dir, work_base_dir):
    abs_file_path = os.path.join(work_base_dir, file_path)
    if not os.path.isfile(abs_file_path):
        raise RuntimeError(f"{os.getcwd()=} {abs_file_path=} must be a file.{file_path=}. {target_dir=} {work_base_dir=}")
    # file_abs_path = os.path.abspath(file_path)
    file_basename = os.path.basename(file_path)

    abs_target_dir = os.path.join(work_base_dir, target_dir)
    relative_path = os.path.relpath(abs_file_path, start=abs_target_dir)
    target_linkfile_path = os.path.join(abs_target_dir, file_basename)
    os.symlink(src=relative_path, dst=target_linkfile_path)
    return target_linkfile_path


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
    # main_flow_dir = os.path.realpath(flow_trigger_dir)

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
#     # main_flow_dir: str
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
#         # template_mixin = next((b for b in bases if isinstance(b, CreateFromTemplateMixinMeta)), None)
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



# class CreateFromTemplateMixin(metaclass=CreateFromTemplateMixinMeta):
MixedInClass_T = TypeVar('MixedInClass_T', bound='CreateFromTemplateMixin')

updates_T = TypeVar('updates_T', bound=Union[BaseModel, Dict[str, Any], Tuple])

class CreateFromTemplateMixin(object): 
    TEMPLATE_DEFAULT_JSON: ClassVar[str] = ""
    TEMPLATE_ADDITIONAL_REQUIRED_FIELDS: ClassVar[Dict[str, Type]] = {}
    # def
    # def __new__(cls, TEMPLATE_DEFAULT_JSON:str, TEMPLATE_ADDITIONAL_REQUIRED_FIELDS:Dict[str, Type]) -> Type:
    @classmethod
    def configure(cls, mixin_cls_name:str, TEMPLATE_DEFAULT_JSON: str, TEMPLATE_ADDITIONAL_REQUIRED_FIELDS: Dict[str, Type]) -> Type['CreateFromTemplateMixin']:
        # cls.TEMPLATE_DEFAULT_JSON = TEMPLATE_DEFAULT_JSON
        # cls.TEMPLATE_ADDITIONAL_REQUIRED_FIELDS = TEMPLATE_ADDITIONAL_REQUIRED_FIELDS
        if not mixin_cls_name.endswith("TemplateMixin"):
             raise ValueError(f'Mixin class name {mixin_cls_name=} must end with "TemplateMixin"')
        # create a new class, inherit `cls`` with class name `mixin_cls_name` and the constants as attr
        configured_mixin_cls:Type['CreateFromTemplateMixin'] = type(mixin_cls_name, (cls,), {
            "TEMPLATE_DEFAULT_JSON":TEMPLATE_DEFAULT_JSON,
            "TEMPLATE_ADDITIONAL_REQUIRED_FIELDS":TEMPLATE_ADDITIONAL_REQUIRED_FIELDS
        })
        return configured_mixin_cls
        # pass
        # raise NotImplementedError(f"{cls.__name__} cannot be instantiated directly. "
        #                         f"It is designed to be used as a mixin class."
        #                         f" And we rewrite its metaclass __call__ method")
    # def __init__(self, TEMPLATE_DEFAULT_JSON:str, TEMPLATE_ADDITIONAL_REQUIRED_FIELDS:Dict[str, Type]) -> Type: # pyright: ignore[reportGeneralTypeIssues]
    #     raise NotImplementedError(f"Cannot be instantiated directly. And we rewrite its metaclass __call__ method")

    @classmethod
    def from_template(cls: Type[MixedInClass_T], updates:Union[BaseModel, Dict[str, Any], Tuple],template_json: Optional[str]=None) -> MixedInClass_T:
        # template_json_to_load = cls.find_template_json_to_load()
        template_data = cls.load_template_data(template_json=template_json)

        if isinstance(updates, BaseModel):
            updates_dict: Dict = updates.model_dump()
        elif isinstance(updates, dict):
            updates_dict: Dict = updates.copy()
        elif isinstance(updates, tuple) and hasattr(updates, '_fields'): # updates is instance of NamedTuple
            updates_dict: Dict = updates._asdict() # pyright: ignore[reportAttributeAccessIssue]
        else:
            raise ValueError(f"call_param Error.cannot convert to a dict {updates=}")
        cls.check_template_required_keys(updates_dict=updates_dict)

        template_data.update(updates_dict)
        if not issubclass(cls, BaseModel):
            raise TypeError(f"cls must be a subclass of pydantic BaseModel {cls=}")
        instance:MixedInClass_T = cls.model_construct(**template_data) # not totally construct. not valid
        instance.model_validate(instance)

        return instance

    @classmethod
    def load_template_data(cls, template_json:Optional[str]=None):
        template_json_to_load = cls.TEMPLATE_DEFAULT_JSON if template_json is None else template_json
        template_json_file_path = os.path.join(DEFAULT_EXAMPLE_DIR, template_json_to_load)
        with open(template_json_file_path, 'r') as f:
            template_data = json.load(f)
        return template_data
    
    @classmethod
    def check_template_required_keys(cls, updates_dict):
        for field, field_type in cls.TEMPLATE_ADDITIONAL_REQUIRED_FIELDS.items():
            if field not in updates_dict:
                raise ValueError(f"Missing template required field: {field}")
            # if not isinstance(updates_dict[field], field_type):
            #     raise TypeError(f"Template {field} must be of {field_type=}. but is {updates_dict[field]=}")

# MyMixinClass = CreateFromTemplateMixin.configure(
#     TEMPLATE_DEFAULT_JSON="hti.json",
#     TEMPLATE_ADDITIONAL_REQUIRED_FIELDS={'a': int})

# class Myclass(BaseModel,
#     CreateFromTemplateMixin(TEMPLATE_DEFAULT_JSON="hti.json", TEMPLATE_ADDITIONAL_REQUIRED_FIELDS={'a', int})):
#     pass


    # @classmethod
    # def create_from(cls, call_param:Union[BaseModel, Dict, Tuple]) -> NodedataType:
    #     self.call_param = call_param
    #     if isinstance(call_param, BaseModel):
    #         update: Dict = call_param.model_dump()
    #     elif isinstance(call_param, dict):
    #         update: Dict = call_param.copy()
    #     elif isinstance(call_param, tuple) and hasattr(call_param, '_fields'): # instance of NamedTuple
    #         update: Dict = dict(call_param)
    #     else:
    #         raise ValueError(f"call_param Error.cannot convert to a dict {call_param=}")
    #     self.check_required_keys(update=update)

    #     self.template_nodedata = self.load_template_nodedata()
    #     self.updated_nodedata = self.template_nodedata.model_copy(update=update)
    #     print(f"note: prepared to validate: {self.updated_nodedata=}")
    #     valid_return:NodedataType = self.updated_nodedata.model_validate(self.updated_nodedata)
    #     print(f"note: valid field pass: self.updated_nodedata as model {valid_return=}")
    #     return valid_return

    # @classmethod
    # def create(cls, data: Union[Dict, Tuple, 'CreateFromTemplateMixin'], template_json: str = None) -> 'TemplateMixin':
    #     if isinstance(data, cls):
    #         return data
    #     elif isinstance(data, dict):
    #         if template_json:
    #             return cls.from_template(template_json, **data)
    #         else:
    #             return cls(**data)
    #     elif isinstance(data, tuple):
    #         return cls(*data)
    #     else:
    #         raise ValueError(f"Unsupported data type: {type(data)}")
        



#%%


class FilesToUploadEntity(BaseModel):
    local_files: List[str]
    fields_in_updated_nodedata: List[str]


class IOHandler(Protocol):
    flow_trigger_dir: str
    main_flow_dir: str
    job_dirname: str
    job_dir: str
    all_produced_paths: defaultdict
    current_produced_paths: List[str] = []

    # def use_job_dir(self, main_flow_dir: str, job_dirname: str) -> str:
    #     raise NotImplementedError
    # def upload_files(self, files_to_upload: FilesToUploadEntity) -> List[str]:
    #     pass
    def use_job_info(self, job_dirname:str) -> Any:
        pass
    def write_pure_file(self, file_path: str, file_content: str) -> str:
        raise NotImplementedError
    def upload_files(self, file_paths: List[str], base_dir: str) -> List[str]:
        raise NotImplementedError
    
    def subdir_context(self, subdirname:str='./') -> Any:
        pass
    # def __enter__(self) -> Any:
    #     pass
    # def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    #     pass


class LocalFileHandler: # implement IOHandler
    
    def __init__(self, flow_trigger_dir: str = "./", main_flow_dir: str = "./") -> None:
        self.flow_trigger_dir = flow_trigger_dir
        self.main_flow_dir = main_flow_dir

        # self.related_workflow = realated_workflow
        # self._job_dir_created: bool = False
        self.all_produced_paths = defaultdict(list)
        self.current_produced_paths: List[str] = []
        self.job_dirname = "default_job/"
        self.job_dir = self.main_flow_dir

    @staticmethod
    def ensure_create_job_dir(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # if not getattr(self, '_job_dir_created', False):
            # if self.job_dir not in self.all_produced_paths:
            if not os.path.isdir(self.job_dir):
                # print(f"creating job_dir:{self.job_dir=} {self.all_produced_paths=}")
                print(f"creating job_dir:{self.job_dir=}")
                self.create_job_dir()
            else:
                # print(f"skip create job_dir {self.job_dir=} {self.all_produced_paths=}")
                pass
            return func(self, *args, **kwargs)
        return wrapper
     
    # @classmethod
    def use_job_info(self, job_dirname: str) -> None:
        # self.main_flow_dir = main_flow_dir
        self.job_dirname = job_dirname
        self.job_dir = os.path.join(self.main_flow_dir, self.job_dirname)

    def create_job_dir(self) -> str:
        self.job_dir = create_path(self.job_dir)
        # self.current_produced_paths.append(self.job_dir)
        self.current_produced_paths = self.all_produced_paths[self.job_dir]
        self.current_produced_paths = list()
        # self._job_dir_created = True
        return self.job_dir
    
    @ensure_create_job_dir
    def write_pure_file(self, file_path: str, file_content: str) -> str:
        abs_file_path = os.path.join(self.job_dir, file_path)
        with open(abs_file_path, 'w') as f:
            f.write(file_content)
        self.current_produced_paths.append(abs_file_path)
        return abs_file_path
    
    @ensure_create_job_dir
    def upload_files(self, file_paths: List[str], base_dir: str) -> List[str]:
        produced_symlinks = self.link_files(link_files=file_paths, base_dir=base_dir)
        return produced_symlinks

    @ensure_create_job_dir
    def link_files(self, link_files: List[str], base_dir: str) -> List[str]:
        produced_symlinks = []
        for file_path in link_files:
            # abs_file_path = os.path.join(self.flow_trigger_dir, file_path)
            target_linkfile_path = create_relative_symlink_file(
                file_path=file_path,
                target_dir=self.job_dir,
                work_base_dir=base_dir
            )
            produced_symlinks.append(target_linkfile_path)
        self.current_produced_paths.extend(produced_symlinks)
        return produced_symlinks
    
    @contextmanager
    def subdir_context(self, subdirname: str = "./") -> Iterator[Any]:
        ori_job_dir = self.job_dir
        try:
            # self.job_dir = job_dir
            self.job_dir = os.path.join(ori_job_dir, subdirname)
            print(f"Entering context {ori_job_dir=} {subdirname=} {self.job_dir=}")
            yield self
        except Exception as e:
            print(f"An exception occurred: {e}")
            raise e
        finally:
            self.job_dir = ori_job_dir
    
    # def __enter__(self) -> 'LocalFileHandler':
    #     # self.job_dirname = job_dirname
    #     # self.job_dir = os.path.join(self.main_flow_dir, self.job_dirname)

    #     print(f"Entering context for {self.main_flow_dir} {self.job_dir}")
    #     return self

    # def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    #     print(f"Exiting context for {self.main_flow_dir} {self.job_dir}")
    #     # 在这里可以进行清理操作，比如关闭文件、释放资源等
    #     if exc_type is not None:
    #         print(f"An exception occurred: {exc_val}")


#%%

class ResultAnalyzer(Protocol):
    pass


class PrefectAnalyzer(): # implement ResultAnalyzer
    pass


#%%


# class AfterPrepare:
#     def __init__(self, upload: bool = True, settings_filename: Optional[str] = 'settings.json'):
#         self.upload = upload
#         self.settings_filename = settings_filename

#     def handle_upload(self, instance: Any) -> None:
#         if self.upload:
#             instance.upload_predefined_files()
    
#     def handle_settings_export(self, instance: Any) -> None:
#         if self.settings_filename:
#             instance.io_handler.write_pure_file(
#                 file_path=self.settings_filename,
#                 file_content=instance.updated_nodedata.model_dump_json(indent=4)
#             )

#     def __call__(self, func: Callable) -> Callable:
#         @wraps(func)
#         def wrapper(instance, *args, **kwargs):
#             # execute origin func
#             result = func(instance, *args, **kwargs)
#             self.handle_upload(instance)
#             self.handle_settings_export(instance)
#             print(f"AfterPrepare:{result=}")
#             return result

#         return wrapper



#%%


@dataclass
class WorkflowService:
    # main_flow_dir: str
    io_handler: IOHandler
    job_executor: JobExecutor
    result_analyzer: ResultAnalyzer


#%%



entity_T = TypeVar('entity_T', bound=BaseModel)
# partial_entity_T = entity_T.model_as_partial()
init_T = TypeVar('init_T', bound=BaseModel)
# call_T = TypeVar('call_T', bound=BaseModel)
# call_T = TypeVar('call_T')
call_T = TypeVar('call_T', bound=Union[BaseModel, Dict[str, Any]])
# call_T = TypeVar('call_T', bound=Union[BaseModel, Dict[str, Any]])
return_T = TypeVar('return_T')

BaseModel_T = TypeVar('BaseModel_T', bound=BaseModel)

class SimulationMetaConfig(BaseModel, extra='allow'):
    # DEFAULT_NODEDATA_CLASS: Type[BaseModel]
    JOB_DIRNAME: str
    DEFAULT_NODEDATA_JSON: str
    # _DEFAULT_
    
    pass
#%%

class InjectionContext:
    _contexts = []

    def __init__(self, injector: Injector):
        self.injector = injector

    @classmethod
    def get_current(cls):
        return cls._contexts[-1] if cls._contexts else None

    def __enter__(self):
        self.__class__._contexts.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__class__._contexts.pop()


@contextmanager
def injection_context(injector: Injector):
    with InjectionContext(injector):
        yield

T = TypeVar('T')

def context_inject(func: Callable[..., T]) -> Callable[..., T]:
    # @inject
    def wrapper(*args, **kwargs):
        context = InjectionContext.get_current()
        print(f"context:{context=}")
        if context:
            injector = context.injector
            # get function parameters typing annotations
            annotations = func.__annotations__
            print(f"context_inject: {annotations=}")
            for param_name, param_type in annotations.items():
                print(f"context_inject: {param_name=}, {param_type=}")
                if param_name not in kwargs and param_name != 'return':
                    if typing.get_origin(param_type) is Union and type(None) in typing.get_args(param_type):
                        # means Optional[Any]. for example: Optional[List[str]],  Union[float, None]  both will not trigger inject
                        pass
                    else:
                        try:
                            # try to get dependency from injector 
                            kwargs[param_name] = injector.get(param_type)
                        except Exception as e:
                            print(f"{param_type=} {dir(param_type)=} {type(param_type)=} ")
                            print(f"context inject fail!  {param_name=} {param_type=} {context=} {injector=} {annotations=}")
                            raise e
                        # pass  # keep original if fail
        return func(*args, **kwargs)
    return wrapper

class InjectableMeta(type):
    pass
#     def __call__(cls, *args, **kwargs):
#         injection_context = InjectionContext.get_current()
#         if injection_context and injection_context.injector:
#             return injection_context.injector.get(cls)
#         return super().__call__(*args, **kwargs)


# class GenericMeta(type):
class SimulationBaseMeta(type):
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace)
        orig_base = cls.__orig_bases__[0]  # pyright: ignore[reportAttributeAccessIssue]
        type_args = get_args(orig_base)
        cls.nodedata_type = type_args[0] # pyright: ignore[reportAttributeAccessIssue]
        cls.init_type = type_args[1] # pyright: ignore[reportAttributeAccessIssue]
        # cls.call_type = type_args[2] # pyright: ignore[reportAttributeAccessIssue]
        cls.return_type = type_args[2] # pyright: ignore[reportAttributeAccessIssue]
        return cls

    # def __instancecheck__(cls, instance):
    #     print(f"__instancecheck__:{cls=} {instance=}")
    #     if cls is BaseModel:
    #         return True # always treated as pydantic BaseModel instance
    #     else:
    #         return isinstance(instance, cls)
# class MockAsBaseModelMeta(type):
#     """Note: introduced for the convenience of pydantic and prefect framework type check.
#     And we implements model_dump method for serialize.
#     to be concrete: prefect/utilities/pydantic.py
#     304-305: method: custom_pydantic_encoder
#     if isinstance(obj, BaseModel):
#                 return obj.model_dump(mode="json")
#     """
#     def __instancecheck__(cls, instance):
#         print(f"__instancecheck__:{cls=} {instance=}")
#         if cls is BaseModel:
#             return True # always treated as pydantic BaseModel instance
#         else:
#             return isinstance(instance, cls)

# class CombinedBaseMeta(SimulationBaseMeta,  type(BaseModel)):
    # def __new__(mcs, name, bases, namespace, **kwargs):
    #     # 首先应用 SimulationBaseMeta 的逻辑
    #     cls = SimulationBaseMeta.__new__(mcs, name, bases, namespace, **kwargs)
    #     # 然后应用 BaseModel 的元类逻辑
    #     # cls = type(BaseModel).__new__(mcs, name, bases, namespace, **kwargs)
    #     return cls
    # pass
# class SimulationBaseMeta(GenericMeta):
    # pass

#%%
# class Mocked(type):
#     # def __instancecheck__(cls, instance):
#     #     print(f"__instancecheck__:{cls=} {instance=}")
#     #     # if cls is BaseModel:
#     #     #     return True # always treated as pydantic BaseModel instance
#     #     # else:
#     #     #     return isinstance(instance, cls)
#     #     return isinstance(instance, cls)

# class A(metaclass=Mocked):
#     pass


# print(isinstance(A(), A))
# print(isinstance(A(), BaseModel))
# print(isinstance(122, A))
# print(isinstance(122, BaseModel))



#%%


entity_T = TypeVar('entity_T', bound=BaseModel)
# partial_entity_T = entity_T.model_as_partial()
# init_T = TypeVar('init_T', bound=BaseModel)
init_T = TypeVar('init_T', bound=Union[BaseModel, Dict[str, Any], NamedTuple])
# call_T = TypeVar('call_T', bound=BaseModel)
# call_T = TypeVar('call_T')
# call_T = TypeVar('call_T', bound=Union[BaseModel, Dict[str, Any]])
# call_T = TypeVar('call_T', bound=Union[BaseModel, Dict[str, Any]])

# nodedata_T = TypeVar('nodedata_T', bound=Union[BaseModel, Dict[str, Any]])
nodedata_T = TypeVar('nodedata_T', bound=BaseModel)
# call_T = TypeVar('call_T', bound=Union[BaseModel, Dict[str, Any]])
return_T = TypeVar('return_T')

#%%
class SimulationBase(Generic[nodedata_T, init_T, return_T],
                    #  BaseModel):
                     metaclass=SimulationBaseMeta):
                    # BaseModel,
                    # metaclass=CombinedBaseMeta):
    
    # must be implement
    DEFAULT_NODEDATA_JSON: str
    JOB_DIRNAME: str
    UPLOAD_LOCAL_FILES: List[str]
    UPLOAD_FIELDS_FILES: List[str]


    # entity_class: ClassVar[Type[entity_T]]  # type: ignore[reportUnknownArgumentType]
    # entity_class: ClassVar[type]
    # entity_class: ClassVar[Type]
    # _type_arg: Type[entity_T]  # pyright: ignore[reportInvalidTypeArguments]

    main_flow_dir: str
    job_dir: str

    # init_entity: Optional[init_T]
    # call_entity: Optional[call_T]
    workflow_service: WorkflowService
    # io_handler:
    
    # default_entity: entity_T
    default_nodedata: nodedata_T
    updated_nodedata: nodedata_T
    # startup_entity: entity_T
    # runtime_entity: entity_T
    runtime_nodedata: nodedata_T
    all_prepared_paths: List[str] = []

    nodedata_type: Type[nodedata_T]
    # nodedata_type: Type[nodedata_T]
    # init_type: Type[init_T]
    # call_type: Type[call_T]
    # return_type: Type[return_T]
    init_type: Type[init_T]
    return_type: Type[return_T]
    # return_DataClass: return_T

    # pyright: ignore[reportInvalidTypeArguments]
    
    # @property
    # @abstractmethod
    # def meta_config(self) -> SimulationMetaConfig:
    #     raise NotImplementedError
    
    # @property
    # @abstractmethod
    # def files_to_upload(self) -> FilesToUploadEntity:
    #     raise NotImplementedError


    # def default_init(self, init_entity: Optional[init_T] = None) -> None:
    #     self.init_entity = init_entity
    #     print("note: init_entity", init_entity)
    #     self.default_entity = self.load_default_entity()
    #     self.startup_entity = self.default_entity.model_copy(
    #         update=(init_entity.model_dump() if init_entity is not None else {}))
    #     print("note: startup_entity", self.startup_entity)
    #     self.main_flow_dir = getattr(self.startup_entity, 'main_flow_dir', default_flow_trigger_dir)
    #     self.job_dir = os.path.join(self.main_flow_dir, self.meta_config.JOB_DIRNAME)


    # def __init__(self, init_entity: Optional[init_T] = None) -> None: 
    #     self.default_init(init_entity=init_entity)

    # @overload
    # def __init__(self) -> NoReturn: ...

    # @overload
    # def __init__(self, workflow_service: WorkflowService) -> None:...

    # @inject

    # def __init__(self, init_data:init_T, setting_update:Dict={}, setting_template=None):
    #     pass

    # def __init__(self, init_param:init_T):
    #     self.init_param = init_param
        
    #     if isinstance(init_param, BaseModel):
    #         update: Dict = init_param.model_dump()
    #     elif isinstance(init_param, dict):
    #         update: Dict = init_param.copy()
    #     elif isinstance(init_param, NamedTuple):
    #         pass
    #     else:
    #         raise ValueError(f"init_param Error.cannot convert to a dict {init_param=} ")
    #     self.default_nodedata = self.load_default_nodedata()
    #     self.updated_nodedata = self.default_nodedata.model_copy(update=update)
    #     print(f"note: prepared to validate: {self.updated_nodedata=}")
    #     valid_return = self.updated_nodedata.model_validate(self.updated_nodedata)
    #     print(f"note: valid field pass: self.updated_nodedata as model {valid_return=}")


    # def 

    # @overload
    # def __call__(self) -> NoReturn: ...

    # @overload
    # def __call__(self, workflow_service: WorkflowService) -> return_T:...

    @context_inject
    def __call__(self, workflow_service: WorkflowService, skip_steps:Optional[List[str]]=None) -> return_T:
        if workflow_service is None:
            raise ValueError("workflow_service must be provided and cannot be None."
                            + "Possible due to Dependency Injection failed"
                            + "")
        self.workflow_service = workflow_service
        self.io_handler = self.workflow_service.io_handler
        self.main_flow_dir = self.io_handler.main_flow_dir
        self.io_handler.use_job_info(job_dirname=self.JOB_DIRNAME)
        self.job_executor = self.workflow_service.job_executor
        self.result_analyzer = self.workflow_service.result_analyzer
        self.job_dir = os.path.join(self.main_flow_dir, self.JOB_DIRNAME)
        
        self.skip_steps = skip_steps

        execute_return: return_T = self.execute(skip_steps=self.skip_steps)  # pyright: ignore[reportCallIssue]  due to Prefect flow decorator
        return execute_return

        
    def for_json(self):
        """Used by simplejson model_dump method for serialize.
        """
        return_dict = {'class_name': self.__class__.__qualname__,
                      'nodadata_type': self.nodedata_type.__qualname__,
                      'updated_nodedata': self.updated_nodedata.model_dump()}
        return return_dict

    # def __init__(self, workflow_service: WorkflowService | None = None) -> None:
    #     if workflow_service is None:
    #         raise ValueError("workflow_service must be provided and cannot be None."
    #                          + "Possible due to Injection failed"
    #                          + ""
    #                          )
    #     self.workflow_service = workflow_service
    #     self.io_handler = self.workflow_service.io_handler

    #     self.main_flow_dir = self.io_handler.main_flow_dir
    #     # with io_hander as handler:
    #     self.io_handler.use_job_info(job_dirname=self.JOB_DIRNAME)
    #     self.job_executor = self.workflow_service.job_executor
    #     self.result_analyzer = self.workflow_service.result_analyzer
    #     self.job_dir = os.path.join(self.main_flow_dir, self.JOB_DIRNAME)


    # def __call__(self, call_entity: call_T) -> return_T:
    #     if isinstance(call_entity, BaseModel):
    #         update: Dict = call_entity.model_dump()
    #     elif isinstance(call_entity, dict):
    #         update: Dict = call_entity
    #     else:
    #         raise ValueError(f"call_entity Error.cannot convert to a dict {call_entity=} ")
    #     self.call_entity = call_entity
    #     self.default_entity = self.load_default_entity()
    #     self.updated_nodedata = self.default_entity.model_copy(update=update)
    #     print("note: prepared to validate: updated_nodedata", self.updated_nodedata)
    #     r_valid = self.updated_nodedata.model_validate(self.updated_nodedata)
    #     print(f"note: valid field pass: updated_nodedata as model {self.updated_nodedata}")
    #     r_execute: return_T = self.execute() # pyright: ignore[reportCallIssue]  due to Prefect flow decorator
    #     return r_execute

    # @flow(persist_result=True)
    @flow
    def execute(self, skip_steps:Optional[List[str]]=None) -> Union[return_T, None]:
        print(f"note: is going to execute job:{self=}")
        # pyright checker ignore reason: Prefect framework provides @task decorator
        # if skip_steps is not None and 'prepare' not in skip_steps:
        self.prepare_return = self.prepare() if 'prepare' not in (skip_steps or []) else None  # pyright: ignore[reportCallIssue]
        print(f"note: execute:{self.prepare_return=}")
        self.run_return = self.run() if 'run' not in (skip_steps or [])  else None # pyright: ignore[reportCallIssue]
        print(f"note: submission hash:{self.run_return=} finished")
        self.extract_return: Union[return_T, None] = self.extract() if 'extract' not in (skip_steps or []) else None # pyright: ignore[reportCallIssue]
        print(f"note: extract data:{self.extract_return=} finished")
        return self.extract_return
    
    # def load_default_nodedata(self) -> Any:
    #     json_path = os.path.join(
    #         DEFAULT_EXAMPLE_DIR,
    #         self.DEFAULT_NODEDATA_JSON)
    #     with open(json_path) as f:
    #         json_dict = json.load(f)
    #         # default_enetity_class: NODEDATA_T = self.meta_config.DEFAULT_NODEDATA_CLASS
    #         # self.meta_config.DEFAULT_NODEDATA_CLASS
    #         simulation_nodedata = self.nodedata_type.model_construct( # 
    #             **json_dict 
    #         )
    #     return simulation_nodedata

    # def upload_predefined_files(self, io_handler: IOHandler) -> List[str]:
    def upload_predefined_files(self,
                                upload_local_files: List[str] = [],
                                upload_fields_files: List[str] = []
                                ) -> List[str]:
        files_symlinks = []

        io_handler = self.io_handler
        r1 = io_handler.upload_files(file_paths=upload_local_files, 
                                     base_dir=io_handler.flow_trigger_dir)
        files_symlinks.extend(r1)

        # r2list = [getattr(self.updated_nodedata, field) for field in upload_fields_files]
        r2list = [getattr(self.updated_nodedata, field) for field in upload_fields_files]
        r2 = io_handler.upload_files(
            file_paths=r2list,
            base_dir=io_handler.flow_trigger_dir
            # file_paths=(getattr(self.startup_entity, field) for field in ["model", "equi_conf"], []))
            )
        files_symlinks.extend(r2)
        return files_symlinks
    
    # @task
    
    @task(cache_key_fn=task_input_json_hash, persist_result=True, refresh_cache=REFRESH_CACHE)
    def prepare(self) -> Any:
        prepare_return = self._prepare()
        return prepare_return

    @abstractmethod
    def _prepare(self) -> Any:
        raise NotImplementedError("Must be override by subclass")


    @task(cache_key_fn=task_input_json_hash, persist_result=True, refresh_cache=REFRESH_CACHE)
    def run(self) -> Any:
        run_return = self._run()
        return run_return
    
    @abstractmethod
    def _run(self):
        raise NotImplementedError("Must be override by subclass")

    # @task
    @task(cache_key_fn=task_input_json_hash, persist_result=True, refresh_cache=REFRESH_CACHE)
    def extract(self) -> return_T:
        extract_return:return_T = self._extract()
        return extract_return
    
    @abstractmethod
    def _extract(self) -> return_T:
        raise NotImplementedError("Must be override by subclass")

#%%







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

class EquiLammpsInput(BaseModel, extra='ignore'):
    # model_config = ConfigDict(str_max_length=10)
    equi_conf: str
    model: str
    mass_map: list = Field(..., validation_alias='model_mass_map')
    nsteps: int
    timestep: float = Field(..., validation_alias='dt')
    ens: str
    temp: float
    pres: float
    tau_t: float
    tau_p: float
    thermo_freq: int = Field(..., validation_alias='stat_freq')
    dump_freq: int
    # stat_skip: int
    # stat_bsize: int
    if_meam: bool
    if_dump_avg_posi: bool
    meam_model: dict

# PartialEquiLammpsInput = EquiLammpsInput.as_partial()

class EquiLammpsAnalyze(BaseModel, extra='ignore'):
    stat_skip: int
    stat_bsize: int

NPTTemplateMixin:Type = CreateFromTemplateMixin.configure(
    mixin_cls_name="NPTTemplateMixin",
    TEMPLATE_DEFAULT_JSON='npt.json',
    TEMPLATE_ADDITIONAL_REQUIRED_FIELDS={
        'temp': float,
        'pres': float,
        'ens': str
    }
)


class EquiLammpsSettings(EquiLammpsInput,
                         EquiLammpsAnalyze):
    pass



class NPTEquiSimulationData(EquiLammpsInput,
                            EquiLammpsAnalyze,
                            NPTTemplateMixin,
                            extra='ignore'):
    pass

# npt_equi_input_data = NPTEquiSimulationData.from_template(updates={})

# EquiLammpsSettings = type('EquiLammpsSettings', (EquiLammpsInput, EquiLammpsAnalyze), {})



#%%
# DEFAULT_EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), '../examples/')
class NPTEquiSimulation(
    SimulationBase[NPTEquiSimulationData,  # nodedata_T,
                   Union[BaseModel, Dict[str, Any], NamedTuple],  # init_T
                   Dict] # return_T
                   ):
    # DEFAULT_NODEDATA_JSON = "npt.json"
    JOB_DIRNAME = "NPT_sim/new_job/"
    UPLOAD_LOCAL_FILES = []
    UPLOAD_FIELDS_FILES = ["model", "equi_conf"]
    NODEDATA_FILENAME = "equi_settings.json"

    nodedata_type:Type[NPTEquiSimulationData] = NPTEquiSimulationData


    # settings_filename
    # NODEDATA_FILENAME = "equi_settings.json"
    # @property
    # def meta_config(self) -> SimulationMetaConfig:
    #     simulation_meta_config = SimulationMetaConfig(
    #         JOB_DIRNAME="NPT_sim/new_job/",
    #         DEFAULT_NODEDATA_JSON="npt.json"
    #     )
    #     return simulation_meta_config

    # @property
    # def files_to_upload(self) -> FilesToUploadEntity:
    #     entity =  FilesToUploadEntity(
    #         local_files=[],
    #         fields_in_updated_nodedata=["model", "equi_conf"],
    #     )
    #     return entity

    # def __init__(self, ):


    def __init__(self, updates={}, template_json:Optional[str]=None):
        self.updates = updates
        self.template_json = template_json
        self.updated_nodedata = self.nodedata_type.from_template(updates=updates, template_json=template_json)
        # print(f"__init__ {updates=}")
        # print(f"__init__ {self.updated_nodedata=}")

    # def model_dump(self, mode="json"): # reportIncompatibleMethodOverride

        # return {}

    # def __init__(self, init_data:NPTEquiSimulationData):
    #     self.init_data = init_data
        # self.init_param = init_param
        
        # if isinstance(init_param, BaseModel):
        #     update: Dict = init_param.model_dump()
        # elif isinstance(init_param, dict):
        #     update: Dict = init_param.copy()
        # elif isinstance(init_param, NamedTuple):
        #     pass
        # else:
        #     raise ValueError(f"init_param Error.cannot convert to a dict {init_param=} ")
        # self.default_nodedata = self.load_default_nodedata()
        # self.updated_nodedata = self.default_nodedata.model_copy(update=update)
        # print(f"note: prepared to validate: {self.updated_nodedata=}")
        # valid_return = self.updated_nodedata.model_validate(self.updated_nodedata)
        # print(f"note: valid field pass: self.updated_nodedata as model {valid_return=}")



    # @task
    # @AfterPrepare(upload=True, settings_filename='equi_settings.json')
    def _prepare(self) -> Dict[str, Any]:
        lammps_input_object = EquiLammpsInput.model_construct(**self.updated_nodedata.model_dump())
        lmp_str = equi.gen_equi_lammps_input(**lammps_input_object.model_dump())

        produced_file = self.io_handler.write_pure_file(
            file_path='in.lammps',
            file_content=lmp_str)

        self.upload_predefined_files(
            upload_local_files=self.UPLOAD_LOCAL_FILES,
            upload_fields_files=self.UPLOAD_FIELDS_FILES,
        )

        self.io_handler.write_pure_file(
            file_path=self.NODEDATA_FILENAME,
            file_content=self.updated_nodedata.model_dump_json(indent=4)
        )

        

        return {"current_produced_paths": self.io_handler.current_produced_paths}
    
    # @task
    def _run(self) -> str:
        submission_hash = self.job_executor.submit(job_dir=self.job_dir)
        return submission_hash

    # @task
    def _extract(self) -> Dict[str, Any]:
        with self.io_handler.subdir_context(subdirname='./') as io_handler:
            info = equi.post_task(io_handler.job_dir)
            result_file_path = os.path.join(io_handler.job_dir, "result.json")
        return info
        # create_link_artifact
#%%

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

NVTTemplateMixin:Type = CreateFromTemplateMixin.configure(
    mixin_cls_name="NVTTemplateMixin",
    TEMPLATE_DEFAULT_JSON='nvt.json',
    TEMPLATE_ADDITIONAL_REQUIRED_FIELDS={
        'temp': float,
        'pres': float,
        'ens': str
    }
)



class NVTEquiSimulationData(EquiLammpsInput,
                            EquiLammpsAnalyze,
                            NVTTemplateMixin,
                            extra='allow'):
    pass


class NVTEquiSimulation(
    SimulationBase[NVTEquiSimulationData,  # nodedata_T,
                   EquiLammpsSettings|Dict[str, Any],  # init_T
                   Dict] # return_T
                   ):
    JOB_DIRNAME = "NVT_sim/new_job/"
    UPLOAD_LOCAL_FILES = []
    UPLOAD_FIELDS_FILES = ["model"]
    NODEDATA_FILENAME = 'equi_settings.json'

    def __init__(self, updates={}, template_json:Optional[str]=None):
        self.updates = updates
        self.template_json = template_json
        self.updated_nodedata = self.nodedata_type.from_template(updates=updates, template_json=template_json)


    # @task
    def _run(self) -> str:
        submission_r = self.job_executor.submit(job_dir=self.job_dir)
        return submission_r
    
    # @task
    def _extract(self) -> Dict[str, Any]:
        with self.io_handler.subdir_context() as io_handler:
            info = equi.post_task(io_handler.job_dir)
            result_file_path = os.path.join(io_handler.job_dir, "result.json")
        return info

    # @task
    # @AfterPrepare(upload=True, settings_filename='equi_settings.json')
    def _prepare(self) -> Dict[str, Any]:
        lammps_input_object = EquiLammpsInput.model_construct(**self.updated_nodedata.model_dump())
        lmp_str = equi.gen_equi_lammps_input(**lammps_input_object.model_dump())
        produced_file = self.io_handler.write_pure_file(
            file_path='in.lammps',
            file_content=lmp_str)

        self.upload_predefined_files(
            upload_local_files=self.UPLOAD_LOCAL_FILES,
            upload_fields_files=self.UPLOAD_FIELDS_FILES,
        )

        self.io_handler.write_pure_file(
            file_path=self.NODEDATA_FILENAME,
            file_content=self.updated_nodedata.model_dump_json(indent=4)
        )

        return {"current_produced_paths": self.io_handler.current_produced_paths}



#%%
def transfer_matching_fields(from_obj: BaseModel, to_type: Type[BaseModel]) -> Dict:
    to_fields = to_type.model_fields
    print(f"{to_fields=}")
    model_fields_set = from_obj.model_fields_set
    if isinstance(model_fields_set, set): # pydantic BaseModel subclass constructor-built object
        from_data = from_obj.model_dump() 
    elif isinstance(model_fields_set, dict): # pydantic BaseModel subclass method model_construct built object
        from_data = model_fields_set.copy()
    else:
        raise ValueError(f"must be a set or dict:{model_fields_set=}"
                         f"debug:{from_obj=} {to_type=}")
    print(f"{from_data=}")

    value_dict = {}

    for field_name,field_info in to_fields.items():
        if field_name in from_data:
            # print(f"1:noalias:{field_name=} {from_data[field_name]=}")
            value_dict[field_name] = from_data[field_name]
        elif isinstance(field_info.validation_alias, str) and field_info.validation_alias in from_data:
            # print(f"2:alias:{field_info.validation_alias=} {from_data[field_info.validation_alias]=}")
            value_dict[field_info.validation_alias] = from_data[field_info.validation_alias]
        elif isinstance(field_info.validation_alias, AliasChoices):
            for alias in field_info.validation_alias.choices:
                if alias in from_data:
                    # value_dict[alias] = from_data[alias]
                    value_dict[field_name] = from_data[alias]
                else:pass
        else:
            pass
    print(f"transfer_matching_fields:{value_dict=}")
    return value_dict
#%%



class HTISimulationSettings(BaseModel, extra='allow', ):
    equi_conf: str
    ncopies: List[int] = Field(..., validation_alias=AliasChoices('ncopies', 'copies'))
    lambda_lj_on: List[str] = Field(...,
        validation_alias=AliasChoices('lambda_angle_on', 'lambda_soft_on'))
    lambda_deep_on: List[str] 
    lambda_spring_off: List[str] = Field(...,
        validation_alias=AliasChoices('lambda_bond_angle_off', 'lambda_soft_off')) # 
    protect_eps: float
    model: str
    mass_map: List[float] = Field(..., validation_alias=AliasChoices('mass_map','model_mass_map'))
    spring_k: float
    soft_param: dict = Field(..., validation_alias=AliasChoices('soft_param','sparam'))
    crystal: str
    langevin: bool
    nsteps: int
    timestep: float = Field(..., validation_alias=AliasChoices('timestep', 'dt'))
    thermo_freq: int = Field(..., validation_alias=AliasChoices('thermo_freq', 'stat_freq'))
    stat_skip: int
    stat_bsize: int
    temp: float
    pres: float
    if_meam: bool = False
    meam_model: Dict|None = None
    ref: str = "vega"
    switch: str = "one-step"



    # @classmethod
    # def load_default_template(cls):
    #     return cls()
    #     pass

HTITemplateMixin:Type = CreateFromTemplateMixin.configure(
    mixin_cls_name='HTITemplateMixin',
    TEMPLATE_DEFAULT_JSON="hti.json",
    TEMPLATE_ADDITIONAL_REQUIRED_FIELDS={})


class HTISimulationNodedata(HTISimulationSettings, HTITemplateMixin):
    accurate_pv_value_from_npt: Optional[str] = None
    accurate_pv_err_value_from_npt: Optional[str] = None
    pass
    # free_energy_value_point:FreeEnergyValuePoint






# class HTIInitData(NamedTuple):
#     extra_input:FreeEnergyValuePoint
#     setting_template:HTISimulationSettings = HTISimulationSettings.load_default_example()
#     setting_update:Dict = {}
#     pass

class HTILammpsInput(BaseModel):
    lamb: float # process control current lambda value
    step: str # process control

    m_spring_k: List[float]
    ens: str # must be `nvt` or `nvt-langevin` controller by langevin
    pres: float # pass in but will not be used
    tau_t: float # not pass in but used
    tau_p: float # not pass in but used

    conf_file: str = Field(..., validation_alias=AliasChoices('conf_file', 'equi_conf'))
    mass_map: List[float] = Field(..., validation_alias=AliasChoices('mass_map', 'model_mass_map'))
    model: str
    nsteps: int
    timestep: float = Field(..., validation_alias=AliasChoices('timestep','dt'))
    temp: float
    thermo_freq: int = Field(..., validation_alias=AliasChoices('thermo_freq','stat_freq'))
    dump_freq: int
    copies: List[int] = Field(..., validation_alias=AliasChoices('copies','ncopies')) # deprecated
    crystal: str
    sparam: dict = Field(..., validation_alias=AliasChoices('sparam','soft_param'))
    switch: str
    if_meam: bool
    meam_model: dict|None

# class FreeEnergyValuePoint(NamedTuple):
class FreeEnergyValuePoint(TypedDict):
    gibbs_free_energy: float # Gibbs free energy[in eV]. 
    gibbs_free_energy_err: float # standard deviation of e1
    temp: float # the e1 corresponding thermo condition
    pres: float # the e1 corresponding thermo condition

# class HTIResultData(NamedTuple):
class HTIResultData(TypedDict):
    p: float
    p_err: float
    v: float
    v_err: float
    e: float # not used here
    e_err: float
    h: float # not used here
    h_err: float
    t: float
    t_err: float
    pv: float # press * volume per atom. [in eV]
    pv_err: float
    free_energy_type: str # usually use Gibbs free energy for TI calculation convenience.
    e0: float # free energy of reference system [in eV]
    de: float # free energy delta during HTI integration path
    de_err: List[float] # (stat_err, inte_err)
    e1: float # Helmholtz free energy of Gibbs free energy [in eV]. Dependents on key `free_energy_type`
    e1_err: float # np.sqrt(de_err[0] ** 2 + pv_err**2)

    free_energy_value_point: FreeEnergyValuePoint


NodedataType = TypeVar('NodedataType', bound=BaseModel)
#%%


# a = {'p':100, 'p_err': 1}

# b = HTIResultData(**a, t=1)
# print(b)

#%%


# @dataclass
# class NodeConfig(Generic[NodedataType]):
#     required_keys_class: Dict[str, type]
#     nodedata_class: type[NodedataType]
#     # default_template: Any
#     template_nodedata_json: str
#     # update:Dict = {}
    
#     @classmethod
#     def from_template(cls):
#         pass
    
    # def __post_init__(self):
#         pass

#     def __call__(self, call_param:Union[BaseModel, Dict, Tuple]) -> NodedataType:
#         self.call_param = call_param
#         if isinstance(call_param, BaseModel):
#             update: Dict = call_param.model_dump()
#         elif isinstance(call_param, dict):
#             update: Dict = call_param.copy()
#         elif isinstance(call_param, tuple) and hasattr(call_param, '_fields'): # instance of NamedTuple
#             update: Dict = dict(call_param)
#         else:
#             raise ValueError(f"call_param Error.cannot convert to a dict {call_param=}")
#         self.check_required_keys(update=update)

#         self.template_nodedata = self.load_template_nodedata()
#         self.updated_nodedata = self.template_nodedata.model_copy(update=update)
#         print(f"note: prepared to validate: {self.updated_nodedata=}")
#         valid_return:NodedataType = self.updated_nodedata.model_validate(self.updated_nodedata)
#         print(f"note: valid field pass: self.updated_nodedata as model {valid_return=}")
#         return valid_return
    
#     def check_required_keys(self, update:Dict[str, Dict]) -> None:
#         for key,value in update.items():
#             KeyClass = self.required_keys_class[key]
#             key_instance = KeyClass(**value)
        

#     def load_template_nodedata(self) -> NodedataType:
#         if not issubclass(self.nodedata_class, BaseModel):
#             raise TypeError("nodedata_class must be a subclass of pydantic.BaseModel")
#         json_path = os.path.join(
#             DEFAULT_EXAMPLE_DIR,
#             self.template_nodedata_json)
#         with open(json_path) as f:
#             json_dict = json.load(f)
#             # default_enetity_class: NODEDATA_T = self.meta_config.DEFAULT_NODEDATA_CLASS
#             # self.meta_config.DEFAULT_NODEDATA_CLASS
#             simulation_nodedata = self.nodedata_class.model_construct( # 
#                 **json_dict 
#             )
#         return simulation_nodedata

# HTIInitFactory = NodeConfig(
#     required_keys_class={'free_energy_value_point':FreeEnergyValuePoint},
#     nodedata_class=HTISimulationSettings,
#     template_nodedata_json='hti.json')

# t:HTISimulationSettings = HTIInitFactory({'free_energy_value_point': {}})


    # update:
# HTINodeConfig = NodeConfigTemplate(
#     required_fields=['free_energy_point']



#%%

# from_obj = HTISimulationSettings.model_construct({'soft_param':{'a':1}, 'equi_conf':"conf.lmp", 'nsteps':10000}) # pyright: ignore[reportCallIssue]

# print(f"{from_obj.model_fields_set=}")

# to_obj = hti_lammps_input.model_construct({})

# value_dict = transfer_matching_fields(from_obj=from_obj, to_type=hti_lammps_input)
# HTISimulationSettings(BaseModel, extra='allow')

#%%
# class test1(BaseModel):
#     equi_conf: str = Field(..., validation_alias=AliasChoices('equi_conf', 'conf_file'))
#     mass_map: List[float] = Field(..., validation_alias=AliasChoices('mass_map', 'model_mass_map', 'another_mass_map'))

# class test2(BaseModel):
#     conf_file: str = Field(..., validation_alias=AliasChoices('conf_file','equi_conf'))
# # t = test1(conf_file='1.txt', mass_map=[2.4,3])

# t1 = test1(equi_conf='1.txt', model_mass_map=[2.4,3]) # pyright: ignore[reportCallIssue]
# # t2 = test2.model_construct({})

# t2 = transfer_matching_fields(from_obj=t1, to_type=test2)


# class HTITasksBase(ABC):
#     @abstractmethod
#     def execute(self, data):
#         pass

# class 


        #     seq_list = ...
        # elif 'switch' == 'two-step':
        #     pass
        # pass


    # def execute(self, *seq_list: List[str]):
    #     switch = self.switch
    #     if switch == "one-step":
    #         all_lambda = parse_seq(jdata["lambda"])
    #     elif switch == "two-step" or switch == "three-step":
    #         if step == "deep_on":
    #             all_lambda = parse_seq(jdata["lambda_deep_on"])
    #         elif step == "spring_off":
    #             all_lambda = parse_seq(jdata["lambda_spring_off"])
    #         elif step == "lj_on":
    #             all_lambda = parse_seq(jdata["lambda_lj_on"])
    #         else:
    #             raise RuntimeError("unknown step", step)
    #         self.seq_list = seq_list
    #     else:
    #         pass

    #     pass


# class OneStepTasks(HTITasksBase):


#     def execute(self, data):
#         self._create_folder_and_write(data[0])

#     def _create_folder_and_write(self, item):
#         # 实现创建文件夹和写入数据的逻辑
#         print(f"Creating folder and writing data for {item}")

# class TwoStepTasks(HTITasksBase):


#     def execute(self, data):
#         for item in data:
#             self._create_folder_and_write(item)

#     def _create_folder_and_write(self, item):
#         # 实现创建文件夹和写入数据的逻辑
#         print(f"Creating folder and writing data for {item}")

# class ThreeStepTasks(HTITasksBase):
#     def execute(self, data):
#         for item in data:
#             self._create_folder_and_write(item)

#     def _create_folder_and_write(self, item):
#         # 实现创建文件夹和写入数据的逻辑
#         print(f"Creating folder and writing data for {item}")

# class HTIFactory:
#     _hti_tasks = {
#         "one-step": OneStepTasks,
#         "two-step": TwoStepTasks,
#         "three-step": ThreeStepTasks
#     }

#     @classmethod
#     def get_hti_tasks(cls, tasks_type):
#         tasks_class = cls._hti_tasks.get(tasks_type)
#         if tasks_class is None:
#             raise ValueError(f"Unknown workflow type: {tasks_type}")
#         return tasks_class()


#%%


class HTISwitch():
    pass

#%%

class HTIIntegraionPath(object):
    def __init__(self, field_name, step_name: str, subtasks_dirname: str):
        # self.field_name = 'lambda_deep_on'
        self.field_name = field_name
        # self.sub_dirname = '01.deep_on'
        self.step_name =  step_name # 'lj_on' # 'deep_on' 'spring_off'
        self.subtasks_dirname = subtasks_dirname

        self.is_parsed = False

        self.seq_list = []
        self.all_lambda = []

    def parse_seq_list(self, seq_list, *, protect_eps:float = 1e-6):
        self.seq_list = seq_list
        self.protect_eps = protect_eps
        self.all_lambda = parse_seq(self.seq_list,
                                    protect_eps=self.protect_eps)
        self.is_parsed = True
        return self.all_lambda
    
    # def use_field_value_by_name(self, ):
    #     field_name = self.field_name
    #     try:
    #         seq_list: List[str] = getattr(, field_name)
    #     except AttributeError as e:
    #         print(f"must provide attribute {field_name=} in {self.updated_nodedata=}")
    #         raise e
    #     integration_path.parse_seq_list(seq_list=seq_list)
    #     return integration_path_list

    def __repr__(self):
        r = (f"working for {self.field_name=}"
            f" {self.step_name=}"
            f" {self.subtasks_dirname=}"
            f" {self.all_lambda=}")
        return r
    
    def generate_subtasks(self, io_handler):
        pass

path_lj_on = HTIIntegraionPath(
    field_name='lambda_lj_on', 
    step_name='lj_on',
    subtasks_dirname='00.lj_on')
# path_deep_on_config = {
#     'field_name': 'lambda_deep_on',
#     'sub_dirname': '01.deep_on'
# }
# path_deep_on = HTIIntegraionPath(**path_deep_on_config)
path_deep_on = HTIIntegraionPath(
    field_name='lambda_deep_on',
    step_name='deep_on',
    subtasks_dirname='01.deep_on',
)

path_spring_off = HTIIntegraionPath(
    field_name='lambda_spring_off',
    step_name='spring_off',
    subtasks_dirname='02.spring_off')

#%%

# class HTISimulationInitData(NamedTuple):
#     known_free_energy_value_point: FreeEnergyValuePoint
#     hti_simulation_settings:HTISimulationSettings = HTISimulationSettings.load_default_example()
#     update_dict: Dict = {}
#     pass


class HTISimulation(
    SimulationBase[HTISimulationNodedata,  # nodedata_T,
                   Union[BaseModel, Dict[str, Any], NamedTuple],  #  init_T
                   HTIResultData] # return_T
                   ):

    # DEFAULT_NODEDATA_JSON = "hti.json"

    JOB_DIRNAME = "HTI_sim/new_job/"
    UPLOAD_LOCAL_FILES = []
    UPLOAD_FIELDS_FILES = ["model"]
    NODEDATA_FILENAME = "in.json"
    # nodedata_type

    def __init__(self, updates={}, template_json:Optional[str]=None):
        self.updates = updates
        self.template_json = template_json
        self.updated_nodedata = self.nodedata_type.from_template(
            updates=updates, template_json=template_json)


    # @task
    # @AfterPrepare(upload=True, settings_filename='hti_settings.json')
    def _prepare(self) -> Dict[str, Any]:
        self.upload_predefined_files(upload_fields_files=self.UPLOAD_FIELDS_FILES)
        switch = self.updated_nodedata.switch
        extra_thermo_info_dict = {
            # 'pres': getattr(self.updated_nodedata, 'pres', 0.0),
            'pres': self.updated_nodedata.pres,
            'temp': self.updated_nodedata.temp,
            'tau_t': getattr(self.updated_nodedata, 'tau_t', 0.1),
            'tau_p': getattr(self.updated_nodedata, 'tau_p', 0.5),
            'dump_freq': getattr(self.updated_nodedata, 'dump_freq', 10000)
        }
        calculated_info_dict = {
            'ens': 'nvt-langevin' if self.updated_nodedata.langevin else 'nvt',
            'm_spring_k': [mass * self.updated_nodedata.spring_k for mass in self.updated_nodedata.mass_map]
        }

        value_dict = transfer_matching_fields(from_obj=self.updated_nodedata,
                                              to_type=HTILammpsInput)
        print(f"{value_dict=}")
        self.integration_path_list = self._choose_integration_path_list(switch=switch)
        for integration_path in self.integration_path_list:
            field_name = integration_path.field_name
            seq_list: List[str] = getattr(self.updated_nodedata, field_name)
            integration_path.parse_seq_list(seq_list=seq_list)

        in_json_dict = ( self.updated_nodedata.model_dump() 
                        | extra_thermo_info_dict
                        | calculated_info_dict )

        self.io_handler.write_pure_file(
            file_path='in.json',
            file_content=json.dumps(obj=in_json_dict, indent=4)
        )
        self._process_integration_path_list(
            integration_path_list=self.integration_path_list,
            in_json_dict=in_json_dict)

        return {"current_produced_paths": self.io_handler.current_produced_paths}

    # @task
    def _run(self) -> str:
        print(f"HTISimulation instance to submit {self.job_dir=}")
        submission_hash = self.job_executor.group_submit(
            job_dir=self.job_dir,
            subtasks_template="./*/task*",
            command="ln -s ../../graph.pb; lmp -i in.lammps"
            )
        # submission_hash = 'Passed!'
        return submission_hash
    
    # @task
    def _extract(self) -> HTIResultData:
        # self.io_handler.use_job_info(job_dirname=self.JOB_DIRNAME)
        accurate_pv_value_from_npt = self.updated_nodedata.accurate_pv_value_from_npt
        accurate_pv_err_value_from_npt = self.updated_nodedata.accurate_pv_err_value_from_npt
        with self.io_handler.subdir_context("./") as io_handler:
            # info = hti.post_tasks(io_handler.job_dir)
            extract_result = hti.compute_task(
                io_handler.job_dir,
                free_energy_type='gibbs', # always use gibbs free energy for the convenience of .
                manual_pv=accurate_pv_value_from_npt,
                manual_pv_err=accurate_pv_err_value_from_npt
                )
            # result_file_path = os.path.join(io_handler.job_dir, "result.json")
        free_energy_value_point = FreeEnergyValuePoint(
            gibbs_free_energy=extract_result['e1'],
            gibbs_free_energy_err=extract_result['e1_err'],
            temp=self.updated_nodedata.temp,
            pres=self.updated_nodedata.pres,
        )
        info = HTIResultData(**extract_result, free_energy_value_point=free_energy_value_point)
        print(f"HTI _extract: {extract_result=} {free_energy_value_point=} {info=}")
        return info

    
    @staticmethod
    def _choose_integration_path_list(switch: str) -> List[HTIIntegraionPath]:
        match switch :
            case 'one-step':
                integration_path_list = [path_deep_on]
            case 'two-step':
                integration_path_list = [path_deep_on, path_spring_off]
            case 'three-step':
                integration_path_list = [path_lj_on, path_deep_on, path_spring_off]
            case _:
                raise ValueError(f"Error option {switch=}")
        return integration_path_list
    
    # def _parse_integration_path_list(self, integration_path_list: List[HTIIntegraionPath]):
    #     for integration_path in integration_path_list:
    #         field_name = integration_path.field_name
    #         try:
    #             seq_list: List[str] = getattr(self.updated_nodedata, field_name)
    #         except AttributeError as e:
    #             print(f"must provide attribute {field_name=} in {self.updated_nodedata=}")
    #             raise e
    #         integration_path.parse_seq_list(seq_list=seq_list)
    #     return integration_path_list

    def _process_integration_path_list(self, integration_path_list: List[Any], 
                                   in_json_dict: Dict[str, Any]
                                   ):
        for path_idx, integration_path in enumerate(integration_path_list):
            print(f"working for {integration_path!r}")
            subtasks_dirname = integration_path.subtasks_dirname
            with self.io_handler.subdir_context(subdirname=subtasks_dirname) as io:
                io.upload_files(file_paths=["out.lmp"], base_dir=os.path.join(self.io_handler.main_flow_dir, self.JOB_DIRNAME))
            for idx, lamb in enumerate(integration_path.all_lambda):
                print(f"{lamb=}")
                lamb_dict = {
                    'step': integration_path.step_name,
                    'lamb': lamb,
                }
                
                hti_lammps_input = HTILammpsInput(**(in_json_dict | lamb_dict))
                lmp_str = hti._gen_lammps_input(**hti_lammps_input.model_dump())
                
                subtask_name = f"{integration_path.subtasks_dirname}/task.{idx:06d}"
                self._write_subtask_files(subtask_name, lmp_str, lamb)
    
    def _write_subtask_files(self, subtask_name: str, lmp_str: str, lamb: float):
        with self.io_handler.subdir_context(subdirname=subtask_name) as io:
            print(f"{subtask_name=}, {io=}")
            io.write_pure_file(file_path='in.lammps', file_content=lmp_str)
            io.write_pure_file(file_path='lambda.out', file_content=str(lamb))
            io.upload_files(
                file_paths=['graph.pb', 'out.lmp'],
                base_dir=os.path.join(self.io_handler.main_flow_dir, self.JOB_DIRNAME)
            )

#%%


# def __init__(self, key):
#         self.key = key
#         MyClass._instances[key] = self  # 在初始化时将实例添加到字典中

#     @classmethod
#     def get_instance(cls, key):
#         return cls._instances.get(key) 
# class 

TIIntegrationPathType = TypeVar('TIIntegrationPathType', bound='TIIntegraionPath')

class TIIntegraionPath(object):
    _instances = {}
    def __init__(self, path:str, point_field_name:str, path_field_name:str, const_thermo_name:str, job_dirname:str):
        self.path = path  # 't' or 'p'
        self.point_field_name = point_field_name
        self.path_field_name = path_field_name
        self.const_thermo_name = const_thermo_name
        self.job_dirname = job_dirname
        self.is_parsed = False
        self.thermo_seq = []
        TIIntegraionPath._instances[path] = self


    def __repr__(self):
        r = (f"working for {self.path=}"
            f" {self.path_field_name=}"
            f" {self.job_dirname=}"
            f" {self.thermo_seq=}")
        return r

    @classmethod
    def get_instance(cls: Type[TIIntegrationPathType], path:str) -> TIIntegrationPathType:
        instance = cls._instances[path]
        return instance
    
    def parse_seq_list(self, thermo_path_seq):
        self.thermo_path_seq = thermo_path_seq
        self.thermo_points_list = list(parse_seq(self.thermo_path_seq,
                                    protect_eps=None))
        self.is_parsed = True
        return self.thermo_points_list
        # temp_list = parse_seq(temp_seq)

t_ti_path = TIIntegraionPath(path='t', point_field_name='temp', path_field_name='temp_seq', const_thermo_name='pres', job_dirname='TI_t_sim')
p_ti_path = TIIntegraionPath(path='p', point_field_name='pres', path_field_name='pres_seq', const_thermo_name='temp', job_dirname='TI_p_sim')


class TISimulationSettings(BaseModel, extra='allow'):
    # conf_file: str = Field(..., validation_alias=AliasChoices('conf_file', 'equi_conf'))
    equi_conf: str = Field(..., validation_alias=AliasChoices('conf_file', 'equi_conf'))
    ncopies: List[int] = Field(default=[1,1,1], validation_alias=AliasChoices('ncopies', 'copies'))
    model: str
    mass_map: List[float] = Field(..., validation_alias=AliasChoices('mass_map','model_mass_map'))
    nsteps: int
    timestep: float
    ens: str
    path: str = Field(..., validation_alias=AliasChoices('path','ti_path'))
    temp_seq: List[str] = Field(..., validation_alias=AliasChoices('temp_seq'))
    pres_seq:  List[str] = Field(..., validation_alias=AliasChoices('pres_seq'))
    temp: Optional[float] = None
    pres: Optional[float] = None
    tau_t: float
    tau_p: float
    thermo_freq: int
    dump_freq: int = 10000
    stat_skip: int
    stat_bsize: int
    if_meam: bool
    meam_model: Optional[Dict[str, Any]] = None


class TILammpsInput(BaseModel):
    # equi_conf: str = Field(..., validation_alias=AliasChoices('conf_file', 'equi_conf'))
    conf_file: str = Field(..., validation_alias=AliasChoices('conf_file', 'equi_conf'))
    mass_map: List[float] = Field(..., validation_alias=AliasChoices('mass_map','model_mass_map'))
    model: str
    nsteps: int
    timestep: float
    ens: str
    temp: int
    pres: float
    tau_t: float
    tau_p: float    
    thermo_freq: int
    dump_freq: int
    copies: List[int] = Field(..., validation_alias=AliasChoices('copies', 'ncopies'))
    if_meam: bool
    meam_model: Optional[Dict[str, Any]] = None
    # lamb: float # process control current lambda value
    # step: str # process control
    # m_spring_k: List[float]
    # ens: str # must be `nvt` or `nvt-langevin` controller by langevin
    # pres: float # pass in but will not be used
    # tau_t: float # not pass in but used
    # tau_p: float # not pass in but used

TITemplateMixin:Type = CreateFromTemplateMixin.configure(
    mixin_cls_name="TITemplateMixin",
    TEMPLATE_DEFAULT_JSON='ti.t.json',
    TEMPLATE_ADDITIONAL_REQUIRED_FIELDS={
        # 'temp': float,
        # 'pres': float,
        # 'ens': str
    }
)


class TISimulationNodeData(
    TISimulationSettings,
    # TILammpsInput,
    TITemplateMixin
    ):
    free_energy_value_point: FreeEnergyValuePoint

#%%
# a = TISimulationNodeData.from_template(updates={})

# print(a)

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





class TISimulation(
    SimulationBase[TISimulationNodeData,  # nodedata_T, 
                   TISimulationSettings|Dict[str, Any],  # init_T
                   Dict] # return_T
                   ):

    # DEFAULT_NODEDATA_JSON = "ti.t.json"
    JOB_DIRNAME = "TI_t_sim/new_job/"
    UPLOAD_LOCAL_FILES = []
    UPLOAD_FIELDS_FILES = ["model"]
    NODEDATA_FILENAME = "ti_settings.json"

    def __init__(self, updates={}, template_json:Optional[str]=None):
        self.updates = updates
        self.template_json = template_json
        self.updated_nodedata = self.nodedata_type.from_template(
            updates=updates, template_json=template_json)



    # @task
    def _prepare(self) -> Dict[str, Any]:
        self.upload_predefined_files(upload_fields_files=self.UPLOAD_FIELDS_FILES)
        path = self.updated_nodedata.path

        ti_integration_path = TIIntegraionPath.get_instance(path=path)
        self.ti_integration_path = ti_integration_path

        thermo_path_seq = getattr(self.updated_nodedata, self.ti_integration_path.path_field_name)
        print(f" {self.ti_integration_path=} {thermo_path_seq=}")

        # self.ti_integration_path.parse_seq_list(
        #     thermo_path_seq=thermo_path_seq
        # )
        # TIIntegraionPath(
        #     path=path,
        #     updated_nodedata.
        # )
        # ti_integration_path.use_thmo
        # thermo_pat
        # thermo_path_seq = getattr(self.updated_nodedata, ti_integration_path.path_field_name)


        # if ti_path == "t":
        thermo_points_list = ti_integration_path.parse_seq_list(thermo_path_seq=thermo_path_seq)

        const_thermo_name = ti_integration_path.const_thermo_name
        extra_thermo_info_dict = {
            const_thermo_name: getattr(self.updated_nodedata, const_thermo_name), 
        }
        print(f"{extra_thermo_info_dict=}")

        # in_json_dict = (self.updated_nodedata.model_dump() | extra_thermo_info_dict)
        in_json_dict = (self.updated_nodedata.model_dump() | extra_thermo_info_dict)

        self.io_handler.write_pure_file(
            file_path=self.NODEDATA_FILENAME,
            file_content=json.dumps(obj=in_json_dict,indent=4)
        )
        # thermo_path = 
        # task_dir = os.path.join(job_abs_dir, "task.%06d" % ii)

        for idx, thermo_point in enumerate(thermo_points_list):
            subtask_name = f"task.{idx:06d}/"
            with self.io_handler.subdir_context(subdirname=subtask_name) as io:
                # print(f"{subtask_name=}, {io=}")

                lammps_input_dict = transfer_matching_fields(
                    from_obj=self.updated_nodedata,
                    to_type=TILammpsInput)

                print(f"{lammps_input_dict=}")
                thermo_point_dict = {
                    ti_integration_path.point_field_name: thermo_point
                }
                lmp_str = ti._gen_lammps_input(**(lammps_input_dict 
                                                  | extra_thermo_info_dict 
                                                  | thermo_point_dict))
                io.write_pure_file(file_path='in.lammps', file_content=lmp_str)
                io.write_pure_file(file_path='thermo.out', file_content=str(thermo_point))
                equi_conf = self.updated_nodedata.equi_conf
                print(f"{equi_conf=}")
                io.upload_files(
                    file_paths=['graph.pb', equi_conf],
                    base_dir=io.flow_trigger_dir
                )
        return {}
                # task_dir = os.path.join(, "task.%06d" % ii)
                # task_abs_dir = create_path(task_dir)
    # @task
    def _run(self) -> str:
        print(f"TISimulation instance to submit {self.job_dir=}")
        # raise RuntimeError
        submission_hash = self.job_executor.group_submit(
            job_dir=self.job_dir,
            subtasks_template="./task*",
            command='ln -s ../graph.pb ./; lmp -i in.lammps')
        # submission_hash = 'Passed!'
        return submission_hash
    
    # @task
    def _extract(self) -> Dict[str, Any]:
        # raise RuntimeError
        # self.io_handler.use_job_info(job_dirname=self.JOB_DIRNAME)
        # Eo = self.updated_nodedata.hti_to_ti_result['']

        with self.io_handler.subdir_context("./") as io_handler:
            # info = hti.post_tasks(io_handler.job_dir)
            Eo = self.updated_nodedata.free_energy_value_point['gibbs_free_energy']
            Eo_err = self.updated_nodedata.free_energy_value_point['gibbs_free_energy_err']
            if self.updated_nodedata.path == 't':
                To = self.updated_nodedata.temp
            elif self.updated_nodedata.path == 'p':
                To = self.updated_nodedata.pres
            else:
                raise RuntimeError(f"Known To value {self.updated_nodedata.path=} {self.updated_nodedata=}")

            extract_result = ti.compute_task(
                job=self.job_dir,
                inte_method='inte',
                Eo=Eo, # free energy value of given reference thermo condition point.
                Eo_err=Eo_err,
                To=To # the known free energy value's thermo conditional.
            )
            # result_file_path = os.path.join(io_handler.job_dir, "result.json")
        return extract_result
        
        # if ti_path == "t":
        # with open(os.path.join(work_base_abs_dir, "ti.t.json")) as j:
        #     ti_jdata = json.load(j)
        #     task_jdata = ti_jdata.copy()
        #     task_jdata["pres"] = start_info["target_pres"]
        #     job_dir = "TI_t_sim"
        # ti_path = start_info["ti_path"]
        pass


#     @task
#     def run(self) -> str:
#         print(f"HTISimulation instance to submit {self.job_dir=}")
#         submission_hash = self.job_executor.group_submit(job_dir=self.job_dir)
#         # submission_hash = 'Passed!'
#         return submission_hash
    
#     @task
#     def extract(self) -> Dict[str, Any]:
#         # self.io_handler.use_job_info(job_dirname=self.JOB_DIRNAME)
#         with self.io_handler.subdir_context("./") as io_handler:
#             # info = hti.post_tasks(io_handler.job_dir)
#             info = hti.compute_task(io_handler.job_dir)
#             # result_file_path = os.path.join(io_handler.job_dir, "result.json")
#         return info


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


class BaseFuncInjectable(metaclass=InjectableMeta):
    @overload
    def __init__(self) -> NoReturn: ...

    @overload
    def __init__(self, io_handler: IOHandler) -> None:...

    @inject
    def __init__(self, io_handler: IOHandler | None = None ) -> None:
                #  npt_dir: Annotated[str, 'npt_dir'],
                #  nvt_dir: Annotated[str, 'nvt_dir']
        if io_handler is None:
            raise ValueError("io_handler must be provided and cannot be None."
                             + "Possible due to Dependency Injection failed"
                             + "")
        self.io_handler = io_handler
        # return self()

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
    #         npt_dir=os.path.join(self.io_handler.main_flow_dir, 
    #                              NPTEquiSimulation.JOB_DIRNAME))
    #     r_lmp = self.io_handler.write_pure_file(file_path="npt_avg.lmp",
    #                                             file_content=npt_avg_conf_lmp)
    #     print(f"header for: npt_avg.lmp:{npt_avg_conf_lmp[0:100]}")
    #     return r_lmp

    # @inject
    # def extract_nvt_to_hti_conf_lmp(self) -> Any:
    #     self.io_handler.use_job_info(job_dirname="HTI_sim/new_job/")
    #     self.io_handler.upload_files(file_paths = [NVTEquiSimulation.JOB_DIRNAME + "/out.lmp"],
    #                                  base_dir=self.io_handler.main_flow_dir)

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
        #     npt_dir=os.path.join(self.io_handler.main_flow_dir,
        #                          NPTEquiSimulation.JOB_DIRNAME))
        # print(f"header for: npt_avg.lmp:{npt_avg_conf_lmp[0:header_print_num]}")
        # return r_lmp
    
    @context_inject
    def __call__(self, io_handler:IOHandler) -> str:
        self.io_handler = io_handler
        self.io_handler.use_job_info(job_dirname=NVTEquiSimulation.JOB_DIRNAME)
        npt_avg_conf_lmp = equi.npt_equi_conf(
            npt_dir=os.path.join(self.io_handler.main_flow_dir,
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
                                     base_dir=self.io_handler.main_flow_dir)
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

def configure_for_testing(binder):
    configuration = Configuration(':localfile:')
    # binder.bind(Configuration, to=configuration, scope=singleton)
    binder.bind(Configuration, to=configuration)

# def configure_for_dirs(binder):
#     binder.bind(Annotated[str, 'npt_dir'], to=NPTEquiSimulation.JOB_DIRNAME)
#     binder.bind(Annotated[str, 'nvt_dir'], to=NVTEquiSimulation.JOB_DIRNAME)


class WorkflowServiceModule(Module):
    def __init__(self, flow_trigger_dir='./', mainflow_dirname="./") -> None:
        self.flow_trigger_dir = flow_trigger_dir
        self.mainflow_dirname = mainflow_dirname
        self.main_flow_dir = os.path.join(self.flow_trigger_dir,
                                               self.mainflow_dirname)

    # def configure(self, binder) -> None:
    #     binder.bind('npt_dir', to=NPTEquiSimulation.JOB_DIRNAME)
    #     binder.bind('nvt_dir', to=NVTEquiSimulation.JOB_DIRNAME)
    # @singleton
    @provider
    def provide_file_handler(self) -> IOHandler:

        local_file_handler = LocalFileHandler(
            flow_trigger_dir=self.flow_trigger_dir,
            main_flow_dir=self.main_flow_dir)
        return local_file_handler
    
    @provider
    def provide_job_executor(self) -> JobExecutor:
        job_executor = DpdispatcherExecutor()
        return job_executor
    
    @provider
    def provide_result_analyzer(self) -> ResultAnalyzer:
        result_analyzer = PrefectAnalyzer()
        return result_analyzer
    
    
    @provider
    def provide_mainflow_dirname(self) -> Annotated[str, 'mainflow_dirname']:
        mainflow_dirname = self.mainflow_dirname
        return mainflow_dirname
    
    @provider
    @inject
    def provide_workflow_service(
        self,
        io_handler: IOHandler,
        job_executor: JobExecutor,
        result_analyzer: ResultAnalyzer
    ) -> WorkflowService:
        return WorkflowService(io_handler=io_handler, 
                                       job_executor=job_executor,
                                       result_analyzer=result_analyzer)

# NPT_DIR_S = NewType('NPT_DIR_S', str)
# NPT_DIR_S = NewType('NPT_DIR_S', str)

# class DirnamesModule(Module):
#     @provider
#     def provide_npt_dir(self) -> NPT_DIR_S:
#         npt_dir = NPTEquiSimulation.JOB_DIRNAME
#         return npt_dir
    
#     @provider
#     def provide_nvt_dir(self) -> Annotated[str, 'nvt_dir']:
#         nvt_dir = NVTEquiSimulation.JOB_DIRNAME
#         return nvt_dir

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

    main_flow_dirname = "free_energy_flow/"
    main_flow_dir = os.path.join(flow_trigger_dir, main_flow_dirname)
    

    workflow_service_module = WorkflowServiceModule(
        flow_trigger_dir=flow_trigger_dir,
        mainflow_dirname=main_flow_dirname)

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
        npt_r = npt(skip_steps=['prepare', 'run']) # want to inject my_injector.get(WorkflowService)

        # npt.io_handler = my_io_handler

        # npt_r = npt()
        accurate_pv_value_from_npt = npt_r['pv']
        accurate_pv_err_value_from_npt = npt_r['pv_err']
        
        pv_dict = {
            'accurate_pv_value_from_npt':accurate_pv_value_from_npt,
            ' accurate_pv_err_value_from_npt': accurate_pv_err_value_from_npt
        }


        # r = NPTResultToNVTConfLmp(header_print_num=100)()


        # nvt = NVTEquiSimulation(updates=( thermo_input._asdict() | {"equi_conf":"npt_avg.lmp", "nsteps": 20000, "stat_bsize": 100}))
        # r = nvt()
        # r = ExtractNVTToHTIConfLmp()()
        hti_sim = HTISimulation(thermo_input._asdict() 
                                | {"nsteps": 5000, "equi_conf": "out.lmp", "ref": "einstein", "switch":"three-step"}
                                | pv_dict)
        # hti_r:HTIResultData = hti_sim()
        hti_r:HTIResultData = hti_sim(skip_steps=['prepare', 'run'])

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
