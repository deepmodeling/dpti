import os
from .base import SimulationBase, CreateFromTemplateMixin
from typing import Type, Union, Dict, Any, NamedTuple, Optional
from pydantic import BaseModel, Field

from ..service.di import InjectionContext, context_inject, injection_context
from ..service.file_handler import IOHandler
from ... import equi



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

class NPTEquiSimulation(
    SimulationBase[NPTEquiSimulationData,  # NodedataType,
                   Union[BaseModel, Dict[str, Any], NamedTuple],  # InitializationType
                   Dict] # ReturnType
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
    SimulationBase[NVTEquiSimulationData,  # NodeDataType,
                   EquiLammpsSettings|Dict[str, Any],  # InitializationType
                   Dict] # ReturnType
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
    