import os
from pathlib import Path
from typing import Type, Union, Dict, Any, NamedTuple, Optional, TypedDict
from pydantic import BaseModel, Field, AliasChoices

from dpti.workflows.flows.base_flow import FlowRuntimeContext
# from ..service.di import InjectionContext, context_inject, injection_context
from dpti.workflows.service.file_handler import IOHandler


from dpti import equi
from dpti.equi import extract as equi_extract
from dpti.workflows.simulations.base import SimulationBase, SettingsBase, workflow_task, FlowRunInfo
from dpti.workflows.simulations.hti_sim import HTISimulation

from dpti.workflows.service.workflow_service_module import BasicWorkflowServices, IOWorkflowServices
from dpti.workflows.service.di import context_inject

from prefect.artifacts import create_link_artifact, create_markdown_artifact
# from dependency_injector import containers, providers
# from dependency_injector.wiring import Provide, inject
# from  

# used for equi.gen_equi_lammps_input as kwargs input
class EquiLammpsInput(BaseModel, extra='ignore'):
    # model_config = ConfigDict(str_max_length=10)
    equi_conf: str = Field(..., validation_alias=AliasChoices('equi_conf', 'conf_lmp'))
    model: str
    mass_map: list = Field(..., validation_alias=AliasChoices('mass_map', 'model_mass_map'))
    nsteps: int
    timestep: float = Field(..., validation_alias=AliasChoices('timestep', 'dt'))
    ens: str
    temp: float
    pres: float
    tau_t: float
    tau_p: float
    thermo_freq: int
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

# NPTTemplateMixin:Type = CreateFromTemplateMixin.configure(
#     mixin_cls_name="NPTTemplateMixin",
#     TEMPLATE_DEFAULT_JSON='npt.json',
#     TEMPLATE_ADDITIONAL_REQUIRED_FIELDS={
#         'temp': float,
#         'pres': float,
#         'ens': str
#     }
# )





# class NPTEquiSimulationNodedata(EquiLammpsInput,
#                             EquiLammpsAnalyze,
#                             NPTTemplateMixin,
#                             extra='ignore'):
#     pass
# class NPTEquiSimulationNodedata():
#     pass


class NPTEquiSimulationSettings(SettingsBase, EquiLammpsInput, EquiLammpsAnalyze):
    # flow_run_info:FlowRunInfo
    if_water: bool = Field(default=False)
    # equi_lammps_input: EquiLammpsInput
    # equi_lammps_analyze: EquiLammpsAnalyze


class NPTEquiSimulationNodedata(BaseModel):
    node_settings: NPTEquiSimulationSettings
    node_upstream_data: Optional[Dict[str, Any]] = None
    node_result: Dict[str, Any] = Field(default_factory=dict)
    pass

    # npt_template: NPTTemplateMixin

# class EquiLammpsSettings(EquiLammpsInput,
#                          EquiLammpsAnalyze):
#     pass

class NPTEquiSimulation(
    SimulationBase[NPTEquiSimulationNodedata,  # NodedataType,
                #    Union[NPTEquiSimulationSettings, Dict[str, Any], NamedTuple],  # SettingsType
                   NPTEquiSimulationSettings,  # SettingsType
                   Dict] # ReturnType
                   ):
    # DEFAULT_NODEDATA_JSON = "npt.json"
    JOB_DIRNAME = "NPT_sim/new_job/"
    UPLOAD_LOCAL_FILES = []
    # UPLOAD_LOCAL_FILES_FIELDS = ["equi_lammps_input.model", "equi_lammps_input.equi_conf"]
    UPLOAD_LOCAL_FILES_FIELDS = ["model", "equi_conf"]
    # NODEDATA_FILENAME = "equi_settings.json"

    nodedata_type:Type[NPTEquiSimulationNodedata] = NPTEquiSimulationNodedata
    # node_settings_type: Type[NPTEquiSimulationSettings] = NPTEquiSimulationSettings

    workflow_services: BasicWorkflowServices


    # settings_filename
    NODEDATA_FILENAME = "equi_settings.json"

    
    # @inject


    # def init_workflow_services(self, workflow_services: BasicWorkflowServices) -> None:
    #     self.io_handler = workflow_services.io_handler
    #     self.job_executor = workflow_services.job_executor
    #     self.report_generator = workflow_services.report_generator
                # self.io_handler.use_job_info(job_dirname=self.JOB_DIRNAME)
        
    @context_inject
    def _init_workflow_services_from_injector(self, workflow_services:BasicWorkflowServices):
        self.workflow_services = workflow_services

    def _prepare(self) -> Dict[str, Any]:

        with self.io_handler.jobdir_context(job_dirname=self.JOB_DIRNAME) as io_handler:
        # io_handler = self.workflow_services.io_handler
        # lammps_input_kwargs_data:EquiLammpsInput = EquiLammpsInput.model_construct(**self.node_settings.model_dump())
            lammps_input_kwargs_data:EquiLammpsInput = EquiLammpsInput.model_construct(
                **self.node_settings.model_dump())
            # lammps_input_kwargs:Dict = self.updated_nodedata.equi_lammps_input.model_dump()
            lmp_str = equi.gen_equi_lammps_input(**lammps_input_kwargs_data.model_dump())

            produced_file = io_handler.write_pure_file(
                file_path='in.lammps',
                file_content=lmp_str)

            self.upload_predefined_files(
                upload_local_files=self.UPLOAD_LOCAL_FILES,
                upload_local_files_fields=self.UPLOAD_LOCAL_FILES_FIELDS,
            )
            self.io_handler.write_pure_file(
                file_path=self.NODEDATA_FILENAME,
                file_content=self.node_settings.model_dump_json(indent=4)
            )

        return {"current_produced_paths": self.io_handler.current_produced_paths}
    
    # @task
    def _run(self) -> str:
        # from dpdispatcher.dlog import dlog
        # dlog.propagate = True 
        submission_hash = self.workflow_services.job_executor.submit(job_dir=self.job_dir)
        return submission_hash

    # @task
    def _extract(self) -> Dict[str, Any]:
        with self.io_handler.jobdir_context(job_dirname=self.JOB_DIRNAME) as io_handler:
            info_dict = equi.post_task(io_handler.job_dir, is_water=self.node_settings.if_water)
            result_file_path = os.path.join(io_handler.job_dir, "result.json")

        print(f"summary NPT to generate markdown: {info_dict=}")
        result_md = equi_summary_md_tmpl.format(info=info_dict)

        create_markdown_artifact(
            key="npt-report",
            markdown=result_md,
            description="NPT Simulation Report",
        )

        return info_dict
    

    def summary(self) -> Dict[str, Any]:
        pass
        # result_file_path = os.path.join(self.io_handler.job_dir, "result.json")

        # with self.io_handler.subdir_context(subdirname='./') as io_handler:
        #     info = equi.post_task(io_handler.job_dir)
        #     result_file_path = os.path.join(io_handler.job_dir, "result.json")
        return {}
        # create_link_artifact
#%%

equi_summary_md_tmpl = \
"""# thermodynamics           value                 err
E        [eV]:  {info[e]:20.8f} {info[e_err]:20.8f}
H        [eV]:  {info[h]:20.8f} {info[h_err]:20.8f}
T         [K]:  {info[t]:20.8f} {info[t_err]:20.8f}
P       [bar]:  {info[p]:20.8f} {info[p_err]:20.8f}
V       [A^3]:  {info[v]:20.8f} {info[v_err]:20.8f}
PV       [eV]:  {info[pv]:20.8f} {info[pv_err]:20.8f}
Lxx       [A]:  {info[lxx]:20.8f} {info[lxx_err]:20.8f}
Lyy       [A]:  {info[lyy]:20.8f} {info[lyy_err]:20.8f}
Lzz       [A]:  {info[lzz]:20.8f} {info[lzz_err]:20.8f}
Lxy       [A]:  {info[lxy]:20.8f} {info[lxy_err]:20.8f}
Lxz       [A]:  {info[lxz]:20.8f} {info[lxz_err]:20.8f}
Lyz       [A]:  {info[lyz]:20.8f} {info[lyz_err]:20.8f}
Pxx     [bar]:  {info[pxx]:20.8f} {info[pxx_err]:20.8f}
Pyy     [bar]:  {info[pyy]:20.8f} {info[pyy_err]:20.8f}
Pzz     [bar]:  {info[pzz]:20.8f} {info[pzz_err]:20.8f}
Pxy     [bar]:  {info[pxy]:20.8f} {info[pxy_err]:20.8f}
Pxz     [bar]:  {info[pxz]:20.8f} {info[pxz_err]:20.8f}
Pyz     [bar]:  {info[pyz]:20.8f} {info[pyz_err]:20.8f}

info:Dict: {info}
"""





#%%

# NVTTemplateMixin:Type = CreateFromTemplateMixin.configure(
#     mixin_cls_name="NVTTemplateMixin",
#     TEMPLATE_DEFAULT_JSON='nvt.json',
#     TEMPLATE_ADDITIONAL_REQUIRED_FIELDS={
#         'temp': float,
#         'pres': float,
#         'ens': str
#     }
# )



# class NVTEquiSimulationData(EquiLammpsInput,
#                             EquiLammpsAnalyze,
#                             NVTTemplateMixin,
#                             extra='allow'):
#     pass






# class NVTEquiSimulationSettings(SettingsBase):
#     equi_lammps_input: EquiLammpsInput
#     equi_lammps_analyze: EquiLammpsAnalyze

class NVTEquiSimulationSettings(SettingsBase, EquiLammpsInput, EquiLammpsAnalyze):
    # flow_run_info:FlowRunInfo
    if_water: bool = Field(default=False)
    # flow_trigger_dir: Optional[str] = Field(default=None)
    # flow_running_dirname: Optional[str] = Field(default=None)

class NVTEquiSimulationUpstreamData(TypedDict):
    avg_conf_from_npt: str


class NVTEquiSimulationNodedata(BaseModel):
    node_settings: NVTEquiSimulationSettings
    node_upstream_data: Optional[NVTEquiSimulationUpstreamData] = None
    node_result: Dict[str, Any] = {}
    pass

class NVTEquiSimulation(
    SimulationBase[NVTEquiSimulationNodedata,  # NodeDataType,
                   NVTEquiSimulationSettings,  # SettingsType
                   Dict] # ReturnType
                   ):
    JOB_DIRNAME = "NVT_sim/new_job/"
    UPLOAD_LOCAL_FILES = []
    # UPLOAD_LOCAL_FILES_FIELDS = ["equi_lammps_input.model", "equi_lammps_input.equi_conf"]
    UPLOAD_LOCAL_FILES_FIELDS = ["model"]
    NODEDATA_FILENAME = 'equi_settings.json'
    settings_data_type: Type[NVTEquiSimulationSettings] = NVTEquiSimulationSettings
    workflow_services: BasicWorkflowServices

    # def __init__(self, updates={}, template_json:Optional[str]=None):
    #     self.updates = updates
    #     self.template_json = template_json
    #     self.updated_nodedata = self.nodedata_type.from_template(updates=updates, template_json=template_json)


    # @task


    # @task
    # @AfterPrepare(upload=True, settings_filename='equi_settings.json')
    def _prepare(self) -> Dict[str, Any]:
        if self.prev_results is not None:
            npt_avg_conf_lmp = self.prev_results['avg_conf_from_npt']
        else:
            npt_avg_conf_lmp = self.node_settings.equi_conf

        npt_avg_conf_lmp_basename = os.path.basename(npt_avg_conf_lmp)
        
        with self.workflow_services.io_handler.jobdir_context(job_dirname=self.JOB_DIRNAME) as io_handler:
            io_handler.upload_file(
                file_path=npt_avg_conf_lmp,
                base_dir=io_handler.flow_running_dir,
            )
            lammps_input_kwargs_data:EquiLammpsInput = EquiLammpsInput.model_construct(
                **(self.node_settings.model_dump()
                | {'equi_conf':npt_avg_conf_lmp_basename})
                )
            
            if not lammps_input_kwargs_data.ens == 'nvt':
                raise ValueError(f"NVT simulation cannot be performed with ens={lammps_input_kwargs_data.ens} must be nvt")
            # lammps_input_kwargs = self.updated_nodedata.equi_lammps_input.model_dump()
            lmp_str = equi.gen_equi_lammps_input(**lammps_input_kwargs_data.model_dump())
            produced_file = io_handler.write_pure_file(
                file_path='in.lammps',
                file_content=lmp_str)

            self.upload_predefined_files(
                upload_local_files=self.UPLOAD_LOCAL_FILES,
                upload_local_files_fields=self.UPLOAD_LOCAL_FILES_FIELDS,
            )

            io_handler.write_pure_file(
                file_path=self.NODEDATA_FILENAME,
                file_content=self.node_settings.model_dump_json(indent=4)
            )

            return {"current_produced_paths": io_handler.current_produced_paths}

    def _run(self) -> str:
        submission_r = self.workflow_services.job_executor.submit(job_dir=self.job_dir)
        return submission_r
    
    # @task
    def _extract(self) -> Dict[str, Any]:
        with self.workflow_services.io_handler.jobdir_context(job_dirname=self.JOB_DIRNAME) as io_handler:
            info = equi.post_task(io_handler.job_dir,
                                  is_water=self.node_settings.if_water)
            result_file_path = os.path.join(io_handler.job_dir, "result.json")

        print(f"summary NVT to generate markdown: {info=}")
        result_md = equi_summary_md_tmpl.format(info=info)

        create_markdown_artifact(
            key="nvt-report",
            markdown=result_md,
            description="NVT Simulation Report",
        )

        return info



#%%


class NPTResultToNVTConfLmp(object):
    def __init__(self, header_print_num: int = 100):
        self.header_print_num = header_print_num
    
    @context_inject
    def __call__(self, workflow_services:IOWorkflowServices) -> str:
        self.workflow_services = workflow_services
        self.io_handler = workflow_services.io_handler
        self.flow_running_dir = self.io_handler.flow_running_dir
        r_lmp = self._run()
        return r_lmp
        
    @workflow_task("run")
    def _run(self) -> str:
        with self.io_handler.jobdir_context(job_dirname=NPTEquiSimulation.JOB_DIRNAME) as io_handler:
            npt_avg_conf_lmp = equi.npt_equi_conf(npt_dir=io_handler.job_dir)
            r_lmp = io_handler.write_pure_file(file_path="npt_avg.lmp",
                                                file_content=npt_avg_conf_lmp)
        return r_lmp
    

    def for_json(self):
        return_dict = {'class_name': self.__class__.__qualname__,
                       'flow_running_dir': self.flow_running_dir}
        return return_dict
    
class ExtractNVTForHTIConfLmp(object):
    def __init__(self, if_dump_avg_posi: bool = False):
        self.if_dump_avg_posi = if_dump_avg_posi

    @context_inject
    def __call__(self, workflow_services:IOWorkflowServices) -> str:
        self.workflow_services = workflow_services
        self.io_handler = self.workflow_services.io_handler
        self.flow_running_dir = self.io_handler.flow_running_dir
        return_file = self._run()
        return return_file

    # def __call__(self,
        

    @workflow_task("run")
    def _run(self) -> str:
        
        nvt_job_dir = os.path.join(self.io_handler.flow_running_dir, NVTEquiSimulation.JOB_DIRNAME)
        hti_job_dir = os.path.join(self.io_handler.flow_running_dir, HTISimulation.JOB_DIRNAME)
        
        if self.if_dump_avg_posi:
            dump_file = "dump.avgposi" # used in equi.extract func
            output_file = "nvt_last_dump_avgposi.lmp"
        else:
            dump_file = "dump.equi"
            output_file = "nvt_last_dump.lmp"
            # rela_target_output_file = os.path.join(NVTEquiSimulation.JOB_DIRNAME, output_file)
            # target_output_file = os.path.join(self.io_handler.job_dir, 
            #                                 rela_target_output_file)
        with self.io_handler.jobdir_context(job_dirname=NVTEquiSimulation.JOB_DIRNAME) as io_handler:
            abs_output_file = os.path.join(io_handler.job_dir, output_file)
            equi_extract(job_dir=nvt_job_dir, output=abs_output_file)
            rela_output_file = os.path.join(io_handler.job_dirname, output_file)
            print(f"ExtractNVTForHTIConfLmp: _run: {rela_output_file=}")

        # ori_nvt_out_lmp = os.path.join(NVTEquiSimulation.JOB_DIRNAME, "out.lmp") # conf_file

        
        # self.io_handler.use_job_info(job_dirname=HTISimulation.JOB_DIRNAME)
        # uploaded_hti_init_conf = self.io_handler.upload_file(file_path=output_file,
        #                              base_dir=nvt_job_dir,
        #                              new_file_name=output_file)
        # print(f"ExtractNVTForHTIConfLmp: _run: upload file: {uploaded_hti_init_conf=}")

        return  rela_output_file
        # return return_file
        # return return_files[0]
    
    def for_json(self):
        return_dict = {'class_name': self.__class__.__qualname__,
                       'flow_running_dir': self.flow_running_dir}
        return return_dict
    
    