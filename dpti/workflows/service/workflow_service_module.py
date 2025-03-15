
#%%
from injector import Module, provider, inject
import os
from typing import Protocol, Annotated, Any
from dpti.workflows.service.file_handler import IOHandler, LocalFileHandler
from dpti.workflows.service.job_executor import JobExecutor, DpdispatcherExecutor
from dpti.workflows.service.report_generator import ReportGenerator, PrefectReportGenerator
# %%
from dataclasses import dataclass
from typing import Any, Protocol

from injector import Module, inject, provider

from .file_handler import IOHandler, LocalFileHandler
from .job_executor import DpdispatcherExecutor, JobExecutor
from .report_generator import PrefectReportGenerator, ReportGenerator

# %%


class WorkflowServices(Protocol):
    io_handler: IOHandler


@dataclass
class IOWorkflowServices(WorkflowServices):
    io_handler: IOHandler


@dataclass
class BasicWorkflowServices(WorkflowServices):
    io_handler: IOHandler
    job_executor: JobExecutor
    report_generator: ReportGenerator


class RayExecutor(Protocol):
    def train(self, job_dir: str, command: str = "") -> Any:
        raise NotImplementedError


@dataclass
class RaytrainerWorkflowServices(WorkflowServices):
    io_handler: IOHandler
    job_executor: JobExecutor
    report_generator: ReportGenerator
    rayjob_trainer: RayExecutor


class WorkflowServiceModule(Module):
    # def __init__(self, flow_trigger_dir='./',
    #              flow_running_dirname="default_flow_running/",
    #              skip_steps:list[str]=[]) -> None:
    #     self.flow_trigger_dir = flow_trigger_dir
    #     self.flow_running_dirname = flow_running_dirname
    #     self.flow_running_dir = os.path.join(self.flow_trigger_dir,
    #                                            self.flow_running_dirname)
    #     self.skip_steps = skip_steps

    # def configure(self, binder) -> None:
    #     binder.bind('npt_dir', to=NPTEquiSimulation.JOB_DIRNAME)
    #     binder.bind('nvt_dir', to=NVTEquiSimulation.JOB_DIRNAME)
    # @singleton

    def __init__(self, flow_trigger_dir: str, flow_running_dirname: str):
        self.flow_trigger_dir = flow_trigger_dir
        self.flow_running_dirname = flow_running_dirname

    @provider
    def provide_file_handler(self) -> IOHandler:
        local_file_handler = LocalFileHandler(
            flow_trigger_dir=self.flow_trigger_dir,
            flow_running_dirname=self.flow_running_dirname,
        )
        return local_file_handler

    @provider
    def provide_job_executor(self) -> JobExecutor:
        job_executor = DpdispatcherExecutor()
        return job_executor

    @provider
    def provide_report_generator(self) -> ReportGenerator:
        report_generator = PrefectReportGenerator()
        return report_generator

    # @provider
    # def provide_flow_running_dirname(self) -> Annotated[str, 'flow_running_dirname']:
    #     flow_running_dirname = self.flow_running_dirname
    #     return flow_running_dirname

    @provider
    @inject
    def provide_basic_workflow_services(
        self,
        io_handler: IOHandler,
        job_executor: JobExecutor,
        report_generator: ReportGenerator,
    ) -> BasicWorkflowServices:
        basic_workflow_services = BasicWorkflowServices(
            io_handler=io_handler,
            job_executor=job_executor,
            report_generator=report_generator,
        )

        return basic_workflow_services

    @provider
    @inject
    def provide_io_workflow_services(self, io_handler: IOHandler) -> IOWorkflowServices:
        io_workflow_services = IOWorkflowServices(io_handler=io_handler)
        return io_workflow_services

    # @provider
    # def provide_flow_runtime_context(self) -> FlowRuntimeContext:
    #     flow_runtime_context = FlowRuntimeContext(
    #         flow_running_dir=self.flow_running_dirname,
    #         flow_trigger_dir=self.flow_trigger_dir,
    #     )
    #     return flow_runtime_context


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
