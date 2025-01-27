import os
import json
from typing import Any, ClassVar, Dict, Tuple, TypeVar, Union, Optional, get_args, Generic, List, Type, NamedTuple, TypedDict
# from typing_extensions import Type, TypedDict
from abc import ABC, abstractmethod

from pydantic import BaseModel, AliasChoices

from ..service.workflow_service_module import WorkflowService
from ..service.di import context_inject

from prefect import flow, task
from ..prefect_task_hash import task_input_json_hash

REFRESH_CACHE = True
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

    flow_running_dir: str
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
    #     self.flow_running_dir = getattr(self.startup_entity, 'flow_running_dir', default_flow_trigger_dir)
    #     self.job_dir = os.path.join(self.flow_running_dir, self.meta_config.JOB_DIRNAME)


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
        self.flow_running_dir = self.io_handler.flow_running_dir
        self.io_handler.use_job_info(job_dirname=self.JOB_DIRNAME)
        self.job_executor = self.workflow_service.job_executor
        self.result_analyzer = self.workflow_service.result_analyzer
        self.job_dir = os.path.join(self.flow_running_dir, self.JOB_DIRNAME)
        
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

    #     self.flow_running_dir = self.io_handler.flow_running_dir
    #     # with io_hander as handler:
    #     self.io_handler.use_job_info(job_dirname=self.JOB_DIRNAME)
    #     self.job_executor = self.workflow_service.job_executor
    #     self.result_analyzer = self.workflow_service.result_analyzer
    #     self.job_dir = os.path.join(self.flow_running_dir, self.JOB_DIRNAME)


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

class FreeEnergyValuePoint(TypedDict):
    gibbs_free_energy: float # Gibbs free energy[in eV]. 
    gibbs_free_energy_err: float # standard deviation of e1
    temp: float # the e1 corresponding thermo condition
    pres: float # the e1 corresponding thermo condition