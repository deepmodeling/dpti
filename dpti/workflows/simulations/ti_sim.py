import os
import json
from typing import List, Dict, TypeVar, TypedDict, Union, Any, Type, NamedTuple, Optional
from  pydantic import BaseModel, Field, AliasChoices
from .base import SimulationBase, CreateFromTemplateMixin, transfer_matching_fields, FreeEnergyValuePoint
from ...lib.utils import parse_seq
from ... import ti



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

class TISimulation(
    SimulationBase[TISimulationNodeData,  # NodeDataType, 
                   TISimulationSettings|Dict[str, Any],  # InitializationType
                   Dict] # ReturnType
                   ):

    # DEFAULT_NODEDATA_JSON = "ti.t.json"
    JOB_DIRNAME = "TI_t_sim/new_job/"
    UPLOAD_LOCAL_FILES = []
    UPLOAD_FIELDS_FILES = ["model", "equi_conf"]
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
                    file_paths=['graph.pb', equi_conf,],
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





#%%
# a = TISimulationNodeData.from_template(updates={})

# print(a)
