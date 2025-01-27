from injector import Module, provider, inject, Annotated
import os
from .file_handler import IOHandler, LocalFileHandler
from .job_executor import JobExecutor, DpdispatcherExecutor
from .result_analyzer import ResultAnalyzer, PrefectAnalyzer
from dataclasses import dataclass



@dataclass
class WorkflowService:
    # flow_running_dir: str
    io_handler: IOHandler
    job_executor: JobExecutor
    result_analyzer: ResultAnalyzer

class WorkflowServiceModule(Module):
    def __init__(self, flow_trigger_dir='./', flow_running_dirname="default_flow_running/") -> None:
        self.flow_trigger_dir = flow_trigger_dir
        self.flow_running_dirname = flow_running_dirname
        self.flow_running_dir = os.path.join(self.flow_trigger_dir,
                                               self.flow_running_dirname)

    # def configure(self, binder) -> None:
    #     binder.bind('npt_dir', to=NPTEquiSimulation.JOB_DIRNAME)
    #     binder.bind('nvt_dir', to=NVTEquiSimulation.JOB_DIRNAME)
    # @singleton
    @provider
    def provide_file_handler(self) -> IOHandler:

        local_file_handler = LocalFileHandler(
            flow_trigger_dir=self.flow_trigger_dir,
            flow_running_dir=self.flow_running_dir)
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
    def provide_flow_running_dirname(self) -> Annotated[str, 'flow_running_dirname']:
        flow_running_dirname = self.flow_running_dirname
        return flow_running_dirname
    
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