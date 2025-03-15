import os
import json
from importlib import resources
# from builtins import AttributeError
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, Tuple, TypeVar, Union, Optional, get_args, Generic, List, Type, NamedTuple, TypedDict
# from typing_extensions import Type, TypedDict
from abc import ABC, abstractmethod
from functools import cached_property


from typing import Dict, Any, Union
import json
import importlib.util
import os
from pathlib import Path
from dataclasses import dataclass
from pydantic import BaseModel, AliasChoices, model_validator, Field, computed_field
# from dependency_injector import containers, providers
# from dependency_injector.wiring import Provide, inject

from dpti.workflows.service.workflow_service_module import WorkflowServices
# from dpti.workflows.service.service_container import WorkflowServices

from dpti.workflows.service.di import context_inject
# from ..service.service_container import WorkflowContainer
from dpti.workflows.service.workflow_service_module import BasicWorkflowServices

from prefect import flow, task
from ..prefect_task_hash import task_input_json_hash
# from dpti.workflows.flows.base_flow import FlowRuntimeContext
from prefect.runtime import flow_run, task_run

# REFRESH_CACHE = True
REFRESH_CACHE = False
DEFAULT_EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), '../../../examples/')




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

entity_T = TypeVar('entity_T', bound=BaseModel)
# partial_entity_T = entity_T.model_as_partial()
SettingsType = TypeVar('SettingsType', bound=BaseModel)
# call_T = TypeVar('call_T', bound=BaseModel)
# call_T = TypeVar('call_T')
call_T = TypeVar('call_T', bound=Union[BaseModel, Dict[str, Any]])
# call_T = TypeVar('call_T', bound=Union[BaseModel, Dict[str, Any]])
ReturnType = TypeVar('ReturnType')

BaseModel_T = TypeVar('BaseModel_T', bound=BaseModel)

class SimulationMetaConfig(BaseModel, extra='allow'):
    # DEFAULT_NODEDATA_CLASS: Type[BaseModel]
    JOB_DIRNAME: str
    DEFAULT_NODEDATA_JSON: str
    # _DEFAULT_
    
    pass
#%%
class FlowRunInfo(BaseModel):
    flow_run_number:Optional[int] = Field(default=0)
    flow_trigger_dir:Optional[str] = Field(default=None)
    flow_running_dirname:Optional[str] = Field(default=None)

class FlowMetaInfo(BaseModel):
    flow_platform: str
    flow_name: str
    flow_version: str



class FlowProcedureControl(BaseModel):
    only_extract: bool = Field(default=False)
    # skip_steps: List[str] = Field(default=[])
    dry_run: bool = Field(default=False)
    debug_mode: bool = Field(default=False)

    @computed_field
    @cached_property
    def skip_steps(self) -> List[str]:
        if self.only_extract:
            skip_steps = ['prepare', 'run']
        else:
            skip_steps = []
        return skip_steps

@dataclass()
class FlowRuntimeContext:
    flow_running_dir: str
    flow_meta_info: FlowMetaInfo
    flow_procedure_control: FlowProcedureControl
    flow_workorder: BaseModel
    

#%%

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

        # for method_name, method in namespace.items():
        #     if hasattr(method, '_workflow_task_name'):
        #         task_name = f"{name}_{method._workflow_task_name}"
        #         decorated_method = task(
        #             name=task_name,
        #             cache_key_fn=task_input_json_hash,
        #             persist_result=True,
        #             refresh_cache=REFRESH_CACHE
        #         )(method)
        #         setattr(cls, method_name, decorated_method)
        # cls._class_name = name
        return cls
#%%

# def workflow_task(method_name: str):
#     def decorator(func):
#         # only label this method, the actual task decorator will be applied in the metaclass
#         func._workflow_task = method_name
#         return func
#     return decorator



#%%


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
def workflow_task(method_name: str, **task_kwargs):
    def decorator(func):
        def wrapper(*args, **kwargs): # args[0] is the instance object(self), like class NPTEquiSimulation's instance.
            if args:
                class_name = args[0].__class__.__name__ # get real class name like, `NPTEquiSimulation`
                task_name = f"{class_name}_{method_name}"
                
                # Check if cache refresh is needed
                instance = args[0]
                force_refresh_tasks = getattr(instance, 'force_refresh_cache', [])
                should_refresh = (
                    REFRESH_CACHE or  # Global refresh setting
                    method_name in force_refresh_tasks  # Task-specific force refresh
                )
            else:
                # Fallback if no instance is provided
                class_name = func.__qualname__.split('.')[0]
                task_name = f"{class_name}_{method_name}"
                should_refresh = REFRESH_CACHE
                
            # Merge default kwargs with custom kwargs
            default_task_kwargs = {
                "name": task_name,
                "task_run_name": lambda: (
                    f"{task_name}-{datetime.now(timezone.utc).strftime('UTC%z_%Y%m%d_%H%M%S_%f')}"
                ),
                "cache_key_fn": task_input_json_hash,
                "persist_result": True,
                "refresh_cache": should_refresh
            }
            # Override defaults with custom task_kwargs
            default_task_kwargs.update(task_kwargs)
            print(f"@workflow_task init: {default_task_kwargs=}")
            
            @task(**default_task_kwargs)
            def task_wrapped(*task_args, **task_kwargs):
                return func(*task_args, **task_kwargs)
                
            return task_wrapped(*args, **kwargs)
        return wrapper
    return decorator



#%%
# @staticmethod
# def workflow_task(method_name: str):
#     def decorator(func):
#         class_name = func.__qualname__.split('.')[0]
#         # actual_class_name = func.__class__._class_name
#         # task_name = f"{actual_class_name}_{method_name}"
#         task_name = f"{class_name}_{method_name}"

#         @task(
#             # name=lambda: f"{flow_run.get_flow_name()}_{method_name}",
#             # name=f"{flow_run.get_flow_name()}_{method_name}",
#             # name=f"{flow_run.get_flow_name()}_{task_run}_{func.__qualname__}_{method_name}",
#             name=task_name,
#             # name=lambda self:self.__class__.__name__ + "_" + method_name,
#             task_run_name=lambda: f"{func.__class__._class_name}_{method_name}",
#             cache_key_fn=task_input_json_hash,
#             persist_result=True,
#             refresh_cache=REFRESH_CACHE
#         )
#         def wrapper(*args, **kwargs):
#             if args:
#                 pass
#                 # print(f"Running {actual_class_name=}_{method_name=}")
#                 # actual_class = args[0].__class__.__name__
#             return func(*args, **kwargs)
#         return wrapper
#     return decorator


#%%


entity_T = TypeVar('entity_T', bound=BaseModel)
# partial_entity_T = entity_T.model_as_partial()
# SettingsType = TypeVar('SettingsType', bound=BaseModel)
SettingsType = TypeVar('SettingsType', bound=BaseModel)
# call_T = TypeVar('call_T', bound=BaseModel)
# call_T = TypeVar('call_T')
# call_T = TypeVar('call_T', bound=Union[BaseModel, Dict[str, Any]])
# call_T = TypeVar('call_T', bound=Union[BaseModel, Dict[str, Any]])

# NodeDataType = TypeVar('NodeDataType', bound=Union[BaseModel, Dict[str, Any]])
NodeDataType = TypeVar('NodeDataType', bound=BaseModel)
# call_T = TypeVar('call_T', bound=Union[BaseModel, Dict[str, Any]])
ReturnType = TypeVar('ReturnType')

#%%
class SimulationBase(Generic[NodeDataType, SettingsType, ReturnType],
                    #  BaseModel):
                     metaclass=SimulationBaseMeta):
                    # BaseModel,
                    # metaclass=CombinedBaseMeta):
    
    # must be implement
    DEFAULT_NODEDATA_JSON: str
    JOB_DIRNAME: str
    UPLOAD_LOCAL_FILES: List[str]
    UPLOAD_LOCAL_FILES_FIELDS: List[str]


    # entity_class: ClassVar[Type[entity_T]]  # type: ignore[reportUnknownArgumentType]
    # entity_class: ClassVar[type]
    # entity_class: ClassVar[Type]
    # _type_arg: Type[entity_T]  # pyright: ignore[reportInvalidTypeArguments]

    flow_running_dir: str
    # job_dir: str

    # init_entity: Optional[SettingsType]
    # call_entity: Optional[call_T]
    # workflow_services: WorkflowServices
    # io_handler:
    
    # default_entity: entity_T
    # default_nodedata: NodeDataType
    # updated_nodedata: NodeDataType
    
    # startup_entity: entity_T
    # runtime_entity: entity_T
    runtime_nodedata: NodeDataType
    all_prepared_paths: List[str] = []

    nodedata_type: Type[NodeDataType]
    node_settings: SettingsType
    # nodedata_type: Type[NodeDataType]
    # init_type: Type[SettingsType]
    # call_type: Type[call_T]
    # return_type: Type[ReturnType]
    init_type: Type[SettingsType]
    return_type: Type[ReturnType]
    # return_DataClass: ReturnType

    # pyright: ignore[reportInvalidTypeArguments]
    
    # @property
    # @abstractmethod
    # def meta_config(self) -> SimulationMetaConfig:
    #     raise NotImplementedError
    
    # @property
    # @abstractmethod
    # def files_to_upload(self) -> FilesToUploadEntity:
    #     raise NotImplementedError


    # def default_init(self, init_entity: Optional[SettingsType] = None) -> None:
    #     self.init_entity = init_entity
    #     print("note: init_entity", init_entity)
    #     self.default_entity = self.load_default_entity()
    #     self.startup_entity = self.default_entity.model_copy(
    #         update=(init_entity.model_dump() if init_entity is not None else {}))
    #     print("note: startup_entity", self.startup_entity)
    #     self.flow_running_dir = getattr(self.startup_entity, 'flow_running_dir', default_flow_trigger_dir)
    #     self.job_dir = os.path.join(self.flow_running_dir, self.meta_config.JOB_DIRNAME)


    # def __init__(self, init_entity: Optional[SettingsType] = None) -> None: 
    #     self.default_init(init_entity=init_entity)

    # @overload
    # def __init__(self) -> NoReturn: ...

    # @overload
    # def __init__(self, workflow_services: WorkflowServices) -> None:...

    # @inject

    # def __init__(self, node_settings:SettingsType, setting_update:Dict={}, setting_template=None):
    #     pass

    # def __init__(self, init_param:SettingsType):
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
    # def __init__(self, node_settings:Union[BaseModel, Dict[str, Any],NamedTuple, None] = None,
    #             updates:Optional[Dict[str, Any]]=None,
    #             template_json:Optional[str]=None,
    #             # workflow_service:Optional[WorkflowService]=None,
    #             # skip_steps:Optional[List[str]]=None
    #             ):
    #     self.node_settings = node_settings
    #     self.updates = updates
    #     self.template_json = template_json
    #     self.nodedata = self.nodedata_type.model_construct(node_settings=self.node_settings)

    #     self._initialize()


        # self.updated_nodedata = self.settings_data_type.from_input(model_data=node_settings, updates=updates, template_json=template_json)

        # self.updated_nodedata = self.nodedata_type.from_input(model_data=node_settings, 
        #                                                       updates=updates,
        #                                                       template_json=template_json)
        # self.updated_nodedata = self.nodedata_type.model_construct(**node_settings)

    # @context_inject
    # def _inject_flow_runtime_context(self, flow_runtime_context:FlowRuntimeContext):
    #     self.flow_runtime_context = flow_runtime_context


    @context_inject
    def __init__(self,
            node_settings:SettingsType,
            # flow_running_dir: str,
            flow_runtime_context:FlowRuntimeContext,
            ):
        self.node_settings = node_settings
        self.flow_running_dir = flow_runtime_context.flow_running_dir
        self.flow_runtime_context = flow_runtime_context
        self.nodedata = self.nodedata_type.model_construct(node_settings=self.node_settings)
        self._initialize()

        self.skip_steps = self.flow_runtime_context.flow_procedure_control.skip_steps

    # def 

    # @overload
    # def __call__(self) -> NoReturn: ...

    # @overload
    # def __call__(self, workflow_service: WorkflowService) -> ReturnType:...

    # @abstractmethod
    def _initialize(self) -> None:
        pass
        # raise NotImplementedError("Must be override by subclass")



    # @context_inject
    # def __call__(self,
    #              node_upstream_data:Optional[Dict[str, Any]] = None,
    #              workflow_service: WorkflowService = None,
    #              skip_steps:Optional[List[str]]=None) -> ReturnType:
    #     if workflow_service is None:
    #         raise ValueError("workflow_service must be provided and cannot be None."
    #                         + "Possible due to Dependency Injection failed"
    #                         + "")
    #     self.node_upstream_data = node_upstream_data

    #     self.workflow_service = workflow_service
    #     self.io_handler = self.workflow_service.io_handler
    #     self.flow_running_dir = self.io_handler.flow_running_dir
    #     self.io_handler.use_job_info(job_dirname=self.JOB_DIRNAME)
    #     self.job_executor = self.workflow_service.job_executor
    #     self.result_analyzer = self.workflow_service.result_analyzer
    #     self.job_dir = os.path.join(self.flow_running_dir, self.JOB_DIRNAME)
        
    #     self.skip_steps = workflow_service.skip_steps if skip_steps is None else skip_steps

    #     # logger.info(f"simulation_base: {self=}")
    #     print(f"simulation_base:__call__: {self.__annotations__=}")
    #     print(f"simulation_base:__call__: {self.node_settings=}")
    #     print(f"simulation_base:__call__: {self.node_upstream_data=}")
    #     print(f"simulation_base:__call__: {self.skip_steps=}")
    #     execute_return: ReturnType = self.execute(skip_steps=self.skip_steps)  # pyright: ignore[reportCallIssue]  due to Prefect flow decorator
    #     return execute_return

        
    def for_json(self):
        """Used by simplejson model_dump method for serialize. (in `task_input_json_hash`)
        """
        return_dict = {'class_name': self.__class__.__qualname__,
                        'flow_running_dir': self.flow_running_dir,
                      'nodedata_type': self.nodedata_type.__qualname__,
                      'node_settings': self.node_settings.model_dump()}
        return return_dict

    # def __init__(self, workflow_service: WorkflowService | None = None) -> None:
    #     if workflow_service is None:
    #         raise ValueError("workflow_service must be provided and cannot be None."
    #                          + "Possible due to Injection failed"
    #                          + ""
    #                          )
    #     self.workflow_service = workflow_service
    #     self.io_handler = self.workflow_service.io_handler

    #     self.flow_running_dir = self.io_handler.flow_running_dir
    #     # with io_hander as handler:
    #     self.io_handler.use_job_info(job_dirname=self.JOB_DIRNAME)
    #     self.job_executor = self.workflow_service.job_executor
    #     self.result_analyzer = self.workflow_service.result_analyzer
    #     self.job_dir = os.path.join(self.flow_running_dir, self.JOB_DIRNAME)

    # @abstractmethod
    # def init_workflow_services(self, workflow_services:WorkflowServices):
    #     pass


    # def prepare_call(self, *args: Any, **kwds: Any) -> Any:


        # return super().__call__(*args, **kwds)
    
    @context_inject
    def _init_workflow_services_from_injector(self, workflow_services:BasicWorkflowServices):
        self.workflow_services = workflow_services

    def _resolve_workflow_services(self, workflow_services:Optional[WorkflowServices]=None):
        if workflow_services is None:
            # use injector to get workflow_services
            self._init_workflow_services_from_injector()
        else:
            self.workflow_services = workflow_services

        if self.workflow_services is None:
            raise ValueError("workflow_services must be provided and cannot be None."
                             + "Possible due to Dependency Injection (via injector package) failed"
                             + "")
        else:
            pass

        self.io_handler = self.workflow_services.io_handler
        self.flow_running_dir = self.io_handler.flow_running_dir
        self.job_dir = os.path.join(self.flow_running_dir, self.JOB_DIRNAME)

    # def _before_call(self, workflow_services:WorkflowServices):
    #     pass

    # def _after_call(self, workflow_services:WorkflowServices):
    #     pass


    # @context_inject
    def __call__(self,
                prev_results:Optional[Dict[str, Any]] = None,
                workflow_services: Optional[WorkflowServices] = None,
                ) -> Union[ReturnType, None]:

        self._resolve_workflow_services(workflow_services=workflow_services)

        self.prev_results = prev_results

        self.job_dir = os.path.join(
            self.io_handler.flow_running_dir,
            self.JOB_DIRNAME)

        print(f"workflow_services: {self.workflow_services=}")
        # 
        # self.io_handler = self.workflow_service.io_handler

        r_execute = self.execute(skip_steps=self.skip_steps)
        # r = self.__call__impl()
        return r_execute

    # def __call__(self, call_entity: call_T) -> ReturnType:
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
    #     r_execute: ReturnType = self.execute() # pyright: ignore[reportCallIssue]  due to Prefect flow decorator
    #     return r_execute

    # @flow(persist_result=True)

    # @property
    # def generate_task_name(self) -> str:
    #     task_name = f"{self.__class__.__name__}_{self.__qualname__}"
    #     return task_name
    
    # @property
    # def generate_flow_name(self) -> str:
    #     flow_name = f"{self.__class__.__name__}_{self.__qualname__}_execute"
    #     return flow_name

    # @flow(name=generate_flow_name)
    def execute(self, skip_steps:Optional[List[str]]=None) -> Union[ReturnType, None]:
        print(f"note: is going to execute job:{self=}")
        # pyright checker ignore reason: Prefect framework provides @task decorator
        # if skip_steps is not None and 'prepare' not in skip_steps:
        self.prepare_return = self.prepare() if 'prepare' not in (skip_steps or []) else None  # pyright: ignore[reportCallIssue]
        print(f"note: execute:{self.prepare_return=}")
        self.run_return = self.run() if 'run' not in (skip_steps or [])  else None # pyright: ignore[reportCallIssue]
        print(f"note: submission hash:{self.run_return=} finished")
        self.extract_return: Union[ReturnType, None] = self.extract() if 'extract' not in (skip_steps or []) else None # pyright: ignore[reportCallIssue]
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

    # def _get_nested_attr(self, obj, field_path: str):
    #     """获取嵌套对象的属性值
        
    #     Args:
    #         obj: 起始对象
    #         field_path: 以点号分隔的字段路径，如 "a.b.c"
    #     """
    #     attrs = field_path.split('.')
    #     value = obj
    #     for attr in attrs:
    #         value = getattr(value, attr)
    #     return value

    def upload_predefined_files(self,
                                upload_local_files: List[str] = [],
                                upload_local_files_fields: List[str] = []
                                ) -> List[str]:
        return_files = []
        io_handler = self.workflow_services.io_handler

        for file_path in upload_local_files:
            return_files_by_direct = io_handler.upload_file(file_path=file_path, 
                                     base_dir=io_handler.flow_trigger_dir)
            return_files.append(return_files_by_direct)

        # r2list = [getattr(self.node_settings, field) for field in upload_files_fields]
        for field in upload_local_files_fields: 
            return_list_by_fields = getattr(self.node_settings, field)
            return_files_by_fields = io_handler.upload_file(
                file_path=return_list_by_fields,
                base_dir=io_handler.flow_trigger_dir
            )
            return_files.append(return_files_by_fields)
        return return_files
    
    # @task

    
    # @task(name="prepare", cache_key_fn=task_input_json_hash, persist_result=True, refresh_cache=REFRESH_CACHE)
    @workflow_task("prepare")
    def prepare(self) -> Any:
        prepare_return = self._prepare()
        return prepare_return

    @abstractmethod
    def _prepare(self) -> Any:
        raise NotImplementedError("Must be override by subclass")


    # @task(name="run", cache_key_fn=task_input_json_hash, persist_result=True, refresh_cache=REFRESH_CACHE)
    @workflow_task("run")
    def run(self) -> Any:
        run_return = self._run()
        return run_return
    
    @abstractmethod
    def _run(self):
        raise NotImplementedError("Must be override by subclass")

    # @task(name="extract", cache_key_fn=task_input_json_hash, persist_result=True, refresh_cache=REFRESH_CACHE)
    @workflow_task("extract")
    def extract(self) -> ReturnType:
        extract_return:ReturnType = self._extract()
        return extract_return
    
    @abstractmethod
    def _extract(self) -> ReturnType:
        raise NotImplementedError("Must be override by subclass")

#%%

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

#%%

def transfer_matching_fields(from_obj: BaseModel, to_type: Type[BaseModel]) -> Dict:
    to_fields = to_type.model_fields
    # print(f"transfer_matching_fields: {to_fields=} {from_obj=} {to_type=}")
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

class FreeEnergyValuePoint(TypedDict):
    gibbs_free_energy: float # Gibbs free energy[in eV]. 
    gibbs_free_energy_err: float # standard deviation of e1
    temp: float # the e1 corresponding thermo condition
    pres: float # the e1 corresponding thermo condition


class SettingsBase(BaseModel, ABC):
    # @model_validator(mode='before')
    # @classmethod 
    # def handle_flat_data(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    #     """同时支持扁平化和嵌套格式的输入"""
    #     if not isinstance(data, dict):
    #         return data
            
    #     nested_fields = cls._get_nested_fields()
    #     if cls._is_nested_format(data, nested_fields):
    #         return data
            
    #     return cls._restructure_flat_data(data, nested_fields)

    # @classmethod
    # def _get_nested_fields(cls) -> Dict[str, Type[BaseModel]]:
    #     """获取所有嵌套的Pydantic模型字段"""
    #     return {
    #         field_name: field_info.annotation
    #         for field_name, field_info in cls.model_fields.items()
    #         if hasattr(field_info.annotation, 'model_fields')
    #     }

    # @classmethod
    # def _is_nested_format(cls, data: Dict[str, Any], nested_fields: Dict[str, Type[BaseModel]]) -> bool:
    #     """检查是否已经是嵌套格式"""
    #     return any(k in data for k in nested_fields)

    # @classmethod
    # def _get_field_mapping(cls, model_class: Type[BaseModel]) -> Dict[str, str]:
    #     """获取字段的所有可能名称到实际字段名的映射"""
    #     mapping = {}
    #     for field_name, field_info in model_class.model_fields.items():
    #         # 添加原始字段名映射
    #         mapping[field_name] = field_name
            
    #         # 处理验证别名
    #         alias = field_info.validation_alias
    #         if alias:
    #             if isinstance(alias, str):
    #                 # 反向映射：别名 -> 实际字段名
    #                 mapping[alias] = field_name
    #             elif isinstance(alias, AliasChoices):
    #                 for alias_choice in alias.choices:
    #                     # 反向映射：别名 -> 实际字段名
    #                     mapping[alias_choice] = field_name
                        
    #     print(f"字段映射 {model_class.__name__}: {mapping}")  # 调试信息
    #     return mapping

    # @classmethod
    # def _restructure_flat_data(cls, data: Dict[str, Any], 
    #                          nested_fields: Dict[str, Type[BaseModel]]) -> Dict[str, Any]:
    #     """重构扁平化数据为嵌套格式"""
    #     nested_data = {}
    #     remaining_data = {}
        
    #     # 为每个嵌套模型创建字段映射
    #     field_mappings = {
    #         field_name: cls._get_field_mapping(model_class)
    #         for field_name, model_class in nested_fields.items()
    #     }
        
    #     # 处理每个输入字段
    #     for key, value in data.items():
    #         field_assigned = False
            
    #         # 检查每个嵌套模型的字段映射
    #         for parent_field, mapping in field_mappings.items():
    #             if key in mapping:
    #                 actual_field = mapping[key]
    #                 nested_data.setdefault(parent_field, {})
    #                 nested_data[parent_field][actual_field] = value
    #                 field_assigned = True
    #                 break
                    
    #         if not field_assigned:
    #             remaining_data[key] = value
                
    #     return remaining_data | nested_data

    # @classmethod
    # def _process_input_data(cls,
    #                      model_data: Union[BaseModel, Dict[str, Any], Tuple, None],
    #                      updates: Optional[Dict[str, Any]] = None,
    #                      template_json: Optional[str] = None) -> Dict[str, Any]:
    #     """处理输入数据"""
    #     # 处理template_json
    #     if template_json:
    #         # with resources.path('dpti.')
    #         # current_file = Path(__file__)
    #         current_file_dir = os.path.dirname(__file__)
    #         # repo_root = current_file_dir.parent.parent.parent  # 回溯到repo根目录
    #         template_json_basedir = os.path.join(current_file_dir, '../', '../', '../' 'examples')
    #         template_json_filepath = os.path.join(template_json_basedir, template_json)
    #         with open(template_json_filepath, 'r') as f:
    #             settings_data = json.load(f)
    #     # 处理model_data
    #     elif isinstance(model_data, BaseModel):
    #         settings_data = model_data.model_dump()
    #     elif isinstance(model_data, dict):
    #         settings_data = model_data.copy()
    #     elif isinstance(model_data, tuple):
    #         settings_data = model_data._asdict()
    #     else:
    #         raise ValueError(f"model_data must be BaseModel/Dict/Tuple type, get {type(model_data)=}")

    #     # 合并updates
    #     if updates:
    #         updated_settings_data = settings_data | updates
    #     else:
    #         updated_settings_data = settings_data.copy()
            
    #     return updated_settings_data

    @classmethod
    def from_input(cls,
                  model_data: Union[BaseModel, Dict[str, Any], Tuple],
                  updates: Optional[Dict[str, Any]] = None,
                  template_json: Optional[str] = None) -> 'SettingsBase':
        """从多种输入数据格式创建实例的工厂方法
        
        Args:
            model_data: 基础数据,可以是BaseModel/Dict/NamedTuple
            updates: 可选的更新数据字典
            template_json: 可选的模板JSON文件路径
            
        Returns:
            SettingsBase实例
        """
        processed_data = cls._process_input_data(
            model_data=model_data,
            updates=updates,
            template_json=template_json
        )

        print(f"-------from_input----{processed_data=}")

        # flattened_data = cls.handle_flat_data(processed_data)
        # print(f"-------from_input----{flattened_data=}")
        
        # 创建实例，这会触发handle_flat_data和其他验证器
        # return cls(**flattened_data)
        return cls.model_validate(processed_data)

    @classmethod 
    def from_template(cls, template_json: str, 
                     updates: Optional[Dict[str, Any]] = None) -> 'SettingsBase':
        """从模板文件创建实例的便捷方法"""
        return cls.from_input(
            model_data={},
            updates=updates,
            template_json=template_json
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any], 
                  updates: Optional[Dict[str, Any]] = None) -> 'SettingsBase':
        """从字典创建实例的便捷方法"""
        return cls.from_input(
            model_data=data,
            updates=updates
        )

    def __init__(self, **data):
        super().__init__(**data)

    # def __init__(self,
    #              model_data: Union[BaseModel, Dict[str, Any], Tuple],
    #              updates: Optional[Dict[str, Any]] = None,
    #              template_json: Optional[str] = None):
        
    #     processed_data = self._process_input_data(
    #         model_data=model_data,
    #         updates=updates, 
    #         template_json=template_json
    #     )
        
    #     # 2. 调用BaseModel.__init__进行验证和初始化
    #     # 这会触发handle_flat_data和其他验证器
    #     super().__init__(**processed_data)
        

    REQUIRED_CLASS_ATTRS:ClassVar[str] = {
        # 'DEFAULT_TEMPLATE_JSON': str,
        # 'NODEDATA_FILENAME': str,
        # 'JOB_DIRNAME': str,
        # 'UPLOAD_LOCAL_FILES': list
    }

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        #check if all required class attributes are defined
        for attr_name, attr_type in cls.REQUIRED_CLASS_ATTRS.items():
            if not hasattr(cls, attr_name):
                raise TypeError(f"Can't instantiate abstract class {cls.__name__} with"
                              f" abstract attribute {attr_name}")
            else:
                pass
            # if not isinstance(getattr(cls, attr_name), attr_type):
            #     raise TypeError(f"Attribute {attr_name} in class {cls.__name__} must be"
            #                   f" of type {attr_type}")

class NodedataBase(BaseModel):
    pass



class ConfigManager:
    """Configuration manager that supports loading from both JSON files and Python modules"""
    
    @staticmethod
    def load_config(config_source: str, flow_trigger_dir: str) -> Dict[str, Any]:
        if config_source.endswith('.json'):
            return ConfigManager._load_json_config(config_source, flow_trigger_dir)
        elif config_source.endswith('.py'):
            return ConfigManager._load_python_config(config_source, flow_trigger_dir)
        else:
            raise ValueError(f"Unsupported config file format: {config_source}")
    
    @staticmethod
    def _load_json_config(json_file: str, flow_trigger_dir: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        config_path = os.path.join(flow_trigger_dir, json_file)
        if not os.path.isfile(config_path):
            raise ValueError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def _load_python_config(py_file: str, flow_trigger_dir: str) -> Dict[str, Any]:
        """
        Load configuration from Python module
        
        The Python config module must define a CONFIG dictionary containing all settings
        """
        config_path = os.path.join(flow_trigger_dir, py_file)
        if not os.path.isfile(config_path):
            raise ValueError(f"Config file not found: {config_path}")
            
        # Dynamically load Python module
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load config module: {config_path}")
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get configuration dictionary
        if hasattr(module, 'CONFIG'):
            return module.CONFIG
        else:
            raise AttributeError(f"Config module must define CONFIG dictionary: {config_path}")
    # @property
    # @abstractmethod
    # def DEFAULT_TEMPLATE_JSON(self) -> Optional[str]:
    #     raise NotImplementedError("Subclasses must define class attribute DEFAULT_TEMPLATE_JSON or set it to None")

    

#%%


