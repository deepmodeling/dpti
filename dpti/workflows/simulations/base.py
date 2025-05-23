import os

# from typing_extensions import Type, TypedDict
from abc import ABC, abstractmethod
from dataclasses import dataclass

# from builtins import AttributeError
from datetime import datetime, timezone
from functools import cached_property
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypedDict,
    TypeVar,
    Union,
    get_args,
)

from prefect import task
from pydantic import AliasChoices, BaseModel, Field, computed_field

from dpti.workflows.prefect.prefect_task_hash import task_input_json_hash
from dpti.workflows.service.di import context_inject
from dpti.workflows.service.workflow_decorator import workflow_task
from dpti.workflows.service.workflow_service_module import (
    BasicWorkflowServices,
    WorkflowServices,
)

# REFRESH_CACHE = True
REFRESH_CACHE = False

# %%

entity_T = TypeVar("entity_T", bound=BaseModel)
SettingsType = TypeVar("SettingsType", bound=BaseModel)
call_T = TypeVar("call_T", bound=Union[BaseModel, Dict[str, Any]])
ReturnType = TypeVar("ReturnType")

BaseModel_T = TypeVar("BaseModel_T", bound=BaseModel)


class SimulationMetaConfig(BaseModel, extra="allow"):
    # DEFAULT_NODEDATA_CLASS: Type[BaseModel]
    JOB_DIRNAME: str
    DEFAULT_NODEDATA_JSON: str


# %%
class FlowRunInfo(BaseModel):
    flow_run_number: Optional[int] = Field(default=0)
    flow_trigger_dir: Optional[str] = Field(default=None)
    flow_running_dirname: Optional[str] = Field(default=None)


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
            skip_steps = ["prepare", "run"]
        else:
            skip_steps = []
        return skip_steps


@dataclass()
class FlowRuntimeContext:
    flow_running_dir: str
    flow_meta_info: FlowMetaInfo
    flow_procedure_control: FlowProcedureControl
    flow_workorder: BaseModel


# %%


# class GenericMeta(type):
class SimulationBaseMeta(type):
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace)
        orig_base = cls.__orig_bases__[0]  # pyright: ignore[reportAttributeAccessIssue]
        type_args = get_args(orig_base)
        cls.nodedata_type = type_args[0]  # pyright: ignore[reportAttributeAccessIssue]
        cls.init_type = type_args[1]  # pyright: ignore[reportAttributeAccessIssue]
        # cls.call_type = type_args[2] # pyright: ignore[reportAttributeAccessIssue]
        cls.return_type = type_args[2]  # pyright: ignore[reportAttributeAccessIssue]

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


# %%
def workflow_task(method_name: str, **task_kwargs):
    def decorator(func):
        def wrapper(
            *args, **kwargs
        ):  # args[0] is the instance object(self), like class NPTEquiSimulation's instance.
            if args:
                class_name = args[
                    0
                ].__class__.__name__  # get real class name like, `NPTEquiSimulation`
                task_name = f"{class_name}_{method_name}"

                # Check if cache refresh is needed
                instance = args[0]
                force_refresh_tasks = getattr(instance, "force_refresh_cache", [])
                should_refresh = (
                    REFRESH_CACHE  # Global refresh setting
                    or method_name in force_refresh_tasks  # Task-specific force refresh
                )
            else:
                # Fallback if no instance is provided
                class_name = func.__qualname__.split(".")[0]
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
                "refresh_cache": should_refresh,
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


# %%


entity_T = TypeVar("entity_T", bound=BaseModel)
SettingsType = TypeVar("SettingsType", bound=BaseModel)

NodeDataType = TypeVar("NodeDataType", bound=BaseModel)
# call_T = TypeVar('call_T', bound=Union[BaseModel, Dict[str, Any]])
ReturnType = TypeVar("ReturnType")


# %%
class SimulationBase(
    Generic[NodeDataType, SettingsType, ReturnType],
    #  BaseModel):
    metaclass=SimulationBaseMeta,
):
    # BaseModel,
    # metaclass=CombinedBaseMeta):

    # must be implement
    DEFAULT_NODEDATA_JSON: str
    JOB_DIRNAME: str
    UPLOAD_LOCAL_FILES: List[str]
    UPLOAD_LOCAL_FILES_FIELDS: List[str]

    flow_running_dir: str
    # job_dir: str

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

    @context_inject
    def __init__(
        self,
        node_settings: SettingsType,
        flow_runtime_context: FlowRuntimeContext,
    ):
        self.node_settings = node_settings
        self.flow_running_dir = flow_runtime_context.flow_running_dir
        self.flow_runtime_context = flow_runtime_context
        self.nodedata = self.nodedata_type.model_construct(
            node_settings=self.node_settings
        )
        self._initialize()

        self.skip_steps = self.flow_runtime_context.flow_procedure_control.skip_steps

    # @overload
    # def __call__(self, workflow_service: WorkflowService) -> ReturnType:...

    # @abstractmethod
    def _initialize(self) -> None:
        pass
        # raise NotImplementedError("Must be override by subclass")

    def for_json(self):
        """Used by simplejson model_dump method for serialize. (in `task_input_json_hash`)."""
        return_dict = {
            "class_name": self.__class__.__qualname__,
            "flow_running_dir": self.flow_running_dir,
            "nodedata_type": self.nodedata_type.__qualname__,
            "node_settings": self.node_settings.model_dump(),
        }
        return return_dict

    @context_inject
    def _init_workflow_services_from_injector(
        self, workflow_services: BasicWorkflowServices
    ):
        self.workflow_services = workflow_services

    def _resolve_workflow_services(
        self, workflow_services: Optional[WorkflowServices] = None
    ):
        if workflow_services is None:
            # use injector to get workflow_services
            self._init_workflow_services_from_injector()
        else:
            self.workflow_services = workflow_services

        if self.workflow_services is None:
            raise ValueError(
                "workflow_services must be provided and cannot be None."
                + "Possible due to Dependency Injection (via injector package) failed"
                + ""
            )
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
    def __call__(
        self,
        prev_results: Optional[Dict[str, Any]] = None,
        workflow_services: Optional[WorkflowServices] = None,
    ) -> Union[ReturnType, None]:
        self._resolve_workflow_services(workflow_services=workflow_services)

        self.prev_results = prev_results

        self.job_dir = os.path.join(self.io_handler.flow_running_dir, self.JOB_DIRNAME)

        print(f"workflow_services: {self.workflow_services=}")
        r_execute = self.execute(skip_steps=self.skip_steps)
        # r = self.__call__impl()
        return r_execute

    def execute(
        self, skip_steps: Optional[List[str]] = None
    ) -> Union[ReturnType, None]:
        print(f"note: is going to execute job:{self=}")
        # pyright checker ignore reason: Prefect framework provides @task decorator
        # if skip_steps is not None and 'prepare' not in skip_steps:
        self.prepare_return = (
            self.prepare() if "prepare" not in (skip_steps or []) else None
        )  # pyright: ignore[reportCallIssue]
        print(f"note: execute:{self.prepare_return=}")
        self.run_return = self.run() if "run" not in (skip_steps or []) else None  # pyright: ignore[reportCallIssue]
        print(f"note: submission hash:{self.run_return=} finished")
        self.extract_return: Union[ReturnType, None] = (
            self.extract() if "extract" not in (skip_steps or []) else None
        )  # pyright: ignore[reportCallIssue]
        print(f"note: extract data:{self.extract_return=} finished")
        return self.extract_return

    def upload_predefined_files(
        self,
        upload_local_files: List[str] = [],
        upload_local_files_fields: List[str] = [],
    ) -> List[str]:
        return_files = []
        io_handler = self.workflow_services.io_handler

        for file_path in upload_local_files:
            return_files_by_direct = io_handler.upload_file(
                file_path=file_path, base_dir=io_handler.flow_trigger_dir
            )
            return_files.append(return_files_by_direct)

        # r2list = [getattr(self.node_settings, field) for field in upload_files_fields]
        for field in upload_local_files_fields:
            return_list_by_fields = getattr(self.node_settings, field)
            return_files_by_fields = io_handler.upload_file(
                file_path=return_list_by_fields, base_dir=io_handler.flow_trigger_dir
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
        extract_return: ReturnType = self._extract()
        return extract_return

    @abstractmethod
    def _extract(self) -> ReturnType:
        raise NotImplementedError("Must be override by subclass")


# %%


def transfer_matching_fields(from_obj: BaseModel, to_type: Type[BaseModel]) -> Dict:
    to_fields = to_type.model_fields
    # print(f"transfer_matching_fields: {to_fields=} {from_obj=} {to_type=}")
    model_fields_set = from_obj.model_fields_set
    if isinstance(
        model_fields_set, set
    ):  # pydantic BaseModel subclass constructor-built object
        from_data = from_obj.model_dump()
    elif isinstance(
        model_fields_set, dict
    ):  # pydantic BaseModel subclass method model_construct built object
        from_data = model_fields_set.copy()
    else:
        raise ValueError(
            f"must be a set or dict:{model_fields_set=}" f"debug:{from_obj=} {to_type=}"
        )
    print(f"{from_data=}")

    value_dict = {}

    for field_name, field_info in to_fields.items():
        if field_name in from_data:
            # print(f"1:noalias:{field_name=} {from_data[field_name]=}")
            value_dict[field_name] = from_data[field_name]
        elif (
            isinstance(field_info.validation_alias, str)
            and field_info.validation_alias in from_data
        ):
            # print(f"2:alias:{field_info.validation_alias=} {from_data[field_info.validation_alias]=}")
            value_dict[field_info.validation_alias] = from_data[
                field_info.validation_alias
            ]
        elif isinstance(field_info.validation_alias, AliasChoices):
            for alias in field_info.validation_alias.choices:
                if str(alias) in from_data:
                    value_dict[field_name] = from_data[str(alias)]
                else:
                    pass
        else:
            pass
    print(f"transfer_matching_fields:{value_dict=}")
    return value_dict


# %%


class FreeEnergyValuePoint(TypedDict):
    gibbs_free_energy: float  # Gibbs free energy[in eV].
    gibbs_free_energy_err: float  # standard deviation of e1
    temp: float  # the e1 corresponding thermo condition
    pres: float  # the e1 corresponding thermo condition


class SettingsBase(BaseModel, ABC):
    pass


class NodedataBase(BaseModel):
    pass


# %%
