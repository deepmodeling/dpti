
import os
import json
import asyncio
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from typing import List, Dict, TypeVar, TypedDict, Union, Any, Type, NamedTuple, Optional
from pydantic import BaseModel, Field, AliasChoices
from dpti.hti import integrate_range_hti
from prefect.artifacts import create_markdown_artifact
from types import SimpleNamespace
#%%
from .base import SimulationBase, CreateFromTemplateMixin, FreeEnergyValuePoint, transfer_matching_fields
from ...lib.utils import parse_seq
from ... import hti, hti_liq, hti_water,hti_liq, hti_ice
from dpti.workflows.service.workflow_service_module import BasicWorkflowServices


# from ...hti_liq import 
#%%

class HTISimulationSettings(BaseModel, extra='allow', ):
    equi_conf: str
    ncopies: Optional[List[int]] = Field(default=[1,1,1], validation_alias=AliasChoices('ncopies', 'copies'))
    lambda_lj_on: Optional[List[str]] = Field(...,
        validation_alias=AliasChoices('lambda_lj_on', 'lambda_angle_on', 'lambda_soft_on'))
    lambda_deep_on: List[str] 
    lambda_spring_off: List[str] = Field(...,
        validation_alias=AliasChoices('lambda_spring_off', 'lambda_bond_angle_off', 'lambda_soft_off')) # 
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
    if_water: bool = Field(default=False)
    if_liquid: bool = Field(default=False)
    if_meam: bool = Field(default=False)
    meam_model: Dict|None = None
    ref: str = "vega"
    switch: str = "one-step"


    # For Ice only
    disorder_corr: Optional[bool] = True # For Ice
    partial_disorder: Optional[str] = None # 5 or 3 for ice5 or ice3
    



    # @classmethod
    # def load_default_template(cls):
    #     return cls()
    #     pass

# HTITemplateMixin:Type = CreateFromTemplateMixin.configure(
#     mixin_cls_name='HTITemplateMixin',
#     TEMPLATE_DEFAULT_JSON="hti.json",
#     TEMPLATE_ADDITIONAL_REQUIRED_FIELDS={})


class HTISimulationUpstreamData(BaseModel):
    conf_file: str
    accurate_pv_value_from_npt: Optional[float] = None
    accurate_pv_err_value_from_npt: Optional[float] = None

    pass

class HTISimulationNodedata(HTISimulationSettings):
    node_settings: HTISimulationSettings
    node_upstream_data: Optional[HTISimulationUpstreamData] = None
    node_result_data: Optional[Dict[str, Any]] = None

    # accurate_pv_value_from_npt: Optional[str] = None
    # accurate_pv_err_value_from_npt: Optional[str] = None
    # pass
    # free_energy_ref_value_point:FreeEnergyValuePoint


# class 

# class HTIInitData(NamedTuple):
#     extra_input:FreeEnergyValuePoint
#     setting_template:HTISimulationSettings = HTISimulationSettings.load_default_example()
#     setting_update:Dict = {}
#     pass

class HTISolidLammpsInput(BaseModel):
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
    copies: Optional[List[int]] = Field(default=None, validation_alias=AliasChoices('copies','ncopies')) # deprecated
    crystal: str
    sparam: dict = Field(..., validation_alias=AliasChoices('sparam','soft_param'))
    switch: str
    if_meam: bool
    meam_model: dict|None

# class FreeEnergyValuePoint(NamedTuple):


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

    free_energy_ref_value_point: FreeEnergyValuePoint


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
#     required_keys_class={'free_energy_ref_value_point':FreeEnergyValuePoint},
#     nodedata_class=HTISimulationSettings,
#     template_nodedata_json='hti.json')

# t:HTISimulationSettings = HTIInitFactory({'free_energy_ref_value_point': {}})


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
class HTIIntegrationPath:
    """concrete HTI integration path instance"""
    def __init__(self, template, seq_list: List[str], protect_eps: float = 1e-6):
        self.template = template
        self.field_name = template.field_name
        self.step_name = template.step_name
        self.subtasks_dirname = template.subtasks_dirname
        self.seq_list = seq_list
        self.protect_eps = protect_eps
        self.all_lambda = list(parse_seq(seq_list, protect_eps=protect_eps))

    # def parse_seq_list(self, seq_list: List[str], protect_eps: float = 1e-6):
    #     self.all_lambda = list(parse_seq(seq_list, protect_eps=protect_eps))
    #     return self.all_lambda

    def __repr__(self):
        r = (f"working for {self.template.field_name=}"
            f" {self.template.step_name=}"
            f" {self.template.subtasks_dirname=}"
            f" {self.seq_list=}"
            f" {self.all_lambda=}")
        return r

class HTIIntegrationPathTemplate:
    """HTI integration path template"""
    def __init__(self, field_name: str, step_name: str, subtasks_dirname: str):
        self.field_name = field_name
        self.step_name = step_name
        self.subtasks_dirname = subtasks_dirname
    
    def __call__(self, seq_list: List[str], protect_eps: float = 1e-6):
        integration_path = HTIIntegrationPath(
            template=self,
            seq_list=seq_list,
            protect_eps=protect_eps
        )
        return integration_path

#%%
class HTILammpsAdapter(ABC):
    """Base adapter class for HTI simulations in LAMMPS"""
    
    @classmethod
    def create(cls, node_settings: HTISimulationSettings) -> 'HTILammpsAdapter':
        """Factory method to create appropriate HTI instance based on input parameters"""
        if_liquid = node_settings.if_liquid
        if_water =  node_settings.if_water

        instance = None
        if if_water and if_liquid:
            instance = WaterHTI(node_settings=node_settings)
        elif if_water and not if_liquid:
            instance = IceHTI(node_settings=node_settings)
        elif not if_water and if_liquid:
            instance = LiquidHTI(node_settings=node_settings)
        else:
            instance = SolidHTI(node_settings=node_settings)   
        return instance
    
    def __init__(self, node_settings: HTISimulationSettings):
        self.node_settings = node_settings

    # @abstractmethod
    # def generate_task(self, input_params: HTILammpsInput) -> None:
    #     """Generate HTI simulation task"""
    #     pass
    @abstractmethod
    def generate_lmp_str(self, input_dict: Dict[str, Any]) -> str:
        """Generate LAMMPS input string"""
        pass
    
    @abstractmethod
    def analyze_task(self, job_dir: str, free_energy_type: str, manual_pv: float, manual_pv_err: float) -> Dict[str, Any]:
        """Analyze HTI simulation results"""
        pass

    @abstractmethod
    def generate_integration_path_list(self, switch: str) -> List[HTIIntegrationPath]:
        """Generate HTI integration path list"""
        pass


# class HTISwitch():
#     pass

#%%

# class HTIIntegraionPath(object):
#     def __init__(self, field_name, step_name: str, subtasks_dirname: str):
#         # self.field_name = 'lambda_deep_on'
#         self.field_name = field_name
#         # self.sub_dirname = '01.deep_on'
#         self.step_name =  step_name # 'lj_on' # 'deep_on' 'spring_off'
#         self.subtasks_dirname = subtasks_dirname

#         self.is_parsed = False

#         self.seq_list = []
#         self.all_lambda = []

#     def parse_seq_list(self, seq_list, *, protect_eps:float = 1e-6):
#         self.seq_list = seq_list
#         self.protect_eps = protect_eps
#         self.all_lambda = parse_seq(self.seq_list,
#                                     protect_eps=self.protect_eps)
#         self.is_parsed = True
#         return self.all_lambda
    
#     # def use_field_value_by_name(self, ):
#     #     field_name = self.field_name
#     #     try:
#     #         seq_list: List[str] = getattr(, field_name)
#     #     except AttributeError as e:
#     #         print(f"must provide attribute {field_name=} in {self.updated_nodedata=}")
#     #         raise e
#     #     integration_path.parse_seq_list(seq_list=seq_list)
#     #     return integration_path_list

#     def __repr__(self):
#         r = (f"working for {self.field_name=}"
#             f" {self.step_name=}"
#             f" {self.subtasks_dirname=}"
#             f" {self.all_lambda=}")
#         return r
    
#     def generate_subtasks(self, io_handler):
#         pass


#%%、





#%%

# class HTISimulationInitData(NamedTuple):
#     free_energy_ref_value_point: FreeEnergyValuePoint
#     hti_simulation_settings:HTISimulationSettings = HTISimulationSettings.load_default_example()
#     update_dict: Dict = {}
#     pass


class HTISimulation(
    SimulationBase[HTISimulationNodedata,  # NodedataType,
                #    Union[BaseModel, Dict[str, Any], NamedTuple],  #  SettingsType
                   HTISimulationSettings, # SettingsType
                   HTIResultData] # ReturnType
                   ):

    # DEFAULT_NODEDATA_JSON = "hti.json"

    JOB_DIRNAME = "HTI_sim/new_job/"
    # UPLOAD_LOCAL_FILES = []
    
    NODEDATA_FILENAME = "in.json"
    settings_data_type: Type[HTISimulationSettings] = HTISimulationSettings
    # settings_data_type: Type[HTISimulationSettings] = HTISimulationSettings
    # nodedata_type
    prev_results: Dict[str, Any] = {}
    hti_lammps_adapter: HTILammpsAdapter
    workflow_services: BasicWorkflowServices


    # def __init__(self, updates={}, template_json:Optional[str]=None):
    #     self.updates = updates
    #     self.template_json = template_json
    #     self.updated_nodedata = self.nodedata_type.from_template(
    #         updates=updates, template_json=template_json)


    # @task
    # @AfterPrepare(upload=True, settings_filename='hti_settings.json')

    def _initialize(self) -> None:
        self.hti_lammps_adapter = HTILammpsAdapter.create(node_settings=self.node_settings)
        self.integration_path_list = self.hti_lammps_adapter.generate_integration_path_list(
            switch=self.node_settings.switch
        )
        


    def _prepare(self) -> Dict[str, Any]:
        with self.workflow_services.io_handler.jobdir_context(job_dirname=self.JOB_DIRNAME) as io_handler:
            self.conf_file = self._get_conf_file()
            self._upload_required_files()
            self.in_json_dict = self._prepare_in_json_dict()

            self.io_handler.write_pure_file(
                file_path='in.json',
            file_content=json.dumps(obj=self.in_json_dict, indent=4)
            )

            self._process_integration_path_list(
                integration_path_list=self.integration_path_list,
                in_json_dict=self.in_json_dict
            )




        
        return {"current_produced_paths": self.io_handler.current_produced_paths}


        if self.node_upstream_data.get('conf_file', None):
            self.conf_file = self.node_upstream_data['conf_file']
        else:
            self.conf_file = self.node_settings.equi_conf
        # conf_f

        switch = self.node_settings.switch
        extra_thermo_info_dict = {
            'equi_conf': 'conf.lmp', # has been linked to the job directory.
            'pres': self.node_settings.pres,
            'temp': self.node_settings.temp,
            'tau_t': getattr(self.node_settings, 'tau_t', 0.1),
            'tau_p': getattr(self.node_settings, 'tau_p', 0.5),
            'dump_freq': getattr(self.node_settings, 'dump_freq', 10000)
        }
        calculated_info_dict = {
            'ens': 'nvt-langevin' if self.node_settings.langevin else 'nvt',
            'm_spring_k': [mass * self.node_settings.spring_k for mass in self.node_settings.mass_map]
        }

        self.io_handler.upload_file(
            file_path=self.conf_file,
            base_dir=self.io_handler.flow_running_dir)

        self.io_handler.upload_file(
            file_path=os.path.basename(self.conf_file),
            base_dir=os.path.join(self.io_handler.flow_running_dir, self.JOB_DIRNAME),
            new_file_name='conf.lmp'
        )


        self.upload_predefined_files(
            upload_local_files=self.UPLOAD_LOCAL_FILES,
            upload_local_files_fields=self.UPLOAD_LOCAL_FILES_FIELDS,
        )



        # value_dict = transfer_matching_fields(from_obj=self.node_settings,
        #                                       to_type=HTILammpsInput)

        # print(f"{value_dict=}")
        self.hti_lammps_adapter = HTILammpsAdapter.create(node_settings=self.node_settings)

        self.integration_path_list = self.hti_lammps_adapter.generate_integration_path_list(
            input_dict=self.node_settings.model_dump()
        )
        for integration_path in self.integration_path_list:
            field_name = integration_path.field_name
            seq_list: List[str] = getattr(self.node_settings, field_name)
            integration_path.parse_seq_list(seq_list=seq_list)

        in_json_dict = ( self.node_settings.model_dump() 
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
        loop = asyncio.get_event_loop()
        try:
            submission_hash = loop.run_until_complete(
                self.workflow_services.job_executor.group_submit(
                    job_dir=self.job_dir,
                    subtasks_template="./*/task*",
                    command='ln -s ../../graph.pb ./; lmp -i in.lammps'
                )
            )
            print(f"_run: submission ends with {submission_hash=}")
        except Exception as e:
            print(f"Error during job execution: {e=}")
            raise e
        return submission_hash
    
    # @task
    def _extract(self) -> HTIResultData:
        # self.io_handler.use_job_info(job_dirname=self.JOB_DIRNAME)
        accurate_pv_value_from_npt = self.prev_results.get('accurate_pv_value_from_npt', None)
        accurate_pv_err_value_from_npt = self.prev_results.get('accurate_pv_err_value_from_npt', None)
        with self.workflow_services.io_handler.jobdir_context(job_dirname=self.JOB_DIRNAME) as io_handler:
            # info = hti.post_tasks(io_handler.job_dir)
            # extract_result = hti.compute_task(
            #     io_handler.job_dir,
            #     free_energy_type='gibbs', # always use gibbs free energy for the convenience of .
            #     manual_pv=accurate_pv_value_from_npt,
            #     manual_pv_err=accurate_pv_err_value_from_npt
            #     )
            extract_result = self.hti_lammps_adapter.analyze_task(
                job_dir=io_handler.job_dir,
                free_energy_type='gibbs',
                manual_pv=accurate_pv_value_from_npt,
                manual_pv_err=accurate_pv_err_value_from_npt
            )
            # result_file_path = os.path.join(io_handler.job_dir, "result.json")
        free_energy_ref_value_point = FreeEnergyValuePoint(
            gibbs_free_energy=extract_result['e1'],
            gibbs_free_energy_err=extract_result['e1_err'],
            temp=self.node_settings.temp,
            pres=self.node_settings.pres,
        )
        info = HTIResultData(**extract_result, free_energy_ref_value_point=free_energy_ref_value_point)
        print(f"HTI _extract: {extract_result=} {free_energy_ref_value_point=} {info=}")


        hti_result_fig_base64_md = get_hti_result_fig_base64_md(
            folder_list=[integration_path.subtasks_dirname for integration_path in self.integration_path_list],
            base_path=os.path.join(self.io_handler.flow_running_dir, self.JOB_DIRNAME)
        )
        hti_summary_md = f"# HTI Summary report\n {extract_result=}\n {info=}\n"

        hti_report_md =  hti_summary_md + "\n" + hti_result_fig_base64_md
        self.io_handler.write_pure_file(file_path='hti_summary.md', file_content=hti_report_md)

        create_markdown_artifact(
            key="hti-report",
            markdown=hti_report_md,
            description="HTI Simulation Report",
        )
        return info

    def _get_conf_file(self) -> str:
        """Get configuration file path from upstream data or node settings"""
        if self.prev_results.get('conf_file', None):
            conf_file = self.prev_results['conf_file']
        else:
            conf_file = self.node_settings.equi_conf
        return conf_file
    
    def _prepare_in_json_dict(self) -> Dict[str, Any]:
        """Prepare thermodynamics and calculation information dictionaries"""
        extra_thermo_info_dict = {
            'equi_conf': 'conf.lmp',  # has been linked to the job directory
            'pres': self.node_settings.pres,
            'temp': self.node_settings.temp,
            'tau_t': getattr(self.node_settings, 'tau_t', 0.1),
            'tau_p': getattr(self.node_settings, 'tau_p', 0.5),
            'dump_freq': getattr(self.node_settings, 'dump_freq', 10000)
        }
        
        calculated_info_dict = {
            'ens': 'nvt-langevin' if self.node_settings.langevin else 'nvt',
            'm_spring_k': [mass * self.node_settings.spring_k 
                        for mass in self.node_settings.mass_map]
        }
        in_json_dict = (self.node_settings.model_dump() 
                | extra_thermo_info_dict 
                | calculated_info_dict)
        return in_json_dict
    
    def _upload_required_files(self) -> None:
        """Upload all required files to flow and job directories"""
        self.upload_predefined_files(
            upload_local_files=[],
            upload_local_files_fields=["model"],
        )
        # Upload config file to flow directory
        self.io_handler.upload_file(
            file_path=self.conf_file,
            base_dir=self.io_handler.flow_running_dir
        )

        # Upload and rename config file to job directory
        self.io_handler.upload_file(
            file_path=os.path.basename(self.conf_file),
            base_dir=os.path.join(self.io_handler.flow_running_dir, self.JOB_DIRNAME),
            new_file_name='conf.lmp'
        )

        # Upload predefined files

        return None
    
        
    # @staticmethod
    # def _choose_integration_path_list(switch: str) -> List[HTIIntegraionPath]:
    #     match switch :
    #         case 'one-step':
    #             integration_path_list = [path_deep_on]
    #         case 'two-step':
    #             integration_path_list = [path_deep_on, path_spring_off]
    #         case 'three-step':
    #             integration_path_list = [path_lj_on, path_deep_on, path_spring_off]
    #         case _:
    #             raise ValueError(f"Error option {switch=}")
    #     return integration_path_list
    
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
            with self.workflow_services.io_handler.subjobdir_context(subjob_dirname=subtasks_dirname) as io: # enter the subtask director like `01.deep_on/`
                io.upload_file(file_path='conf.lmp',
                               base_dir=os.path.join(self.io_handler.flow_running_dir, self.JOB_DIRNAME),
                            #    new_file_name='conf.lmp')
                )

                io.upload_file(file_path=self.node_settings.model,
                               base_dir=os.path.join(self.io_handler.flow_running_dir, self.JOB_DIRNAME),
                               new_file_name='graph.pb')
                
                io.upload_file(file_path='in.json',
                               base_dir=os.path.join(self.io_handler.flow_running_dir, self.JOB_DIRNAME)
                            #    new_file_name='in.json')
                )
            for idx, lamb in enumerate(integration_path.all_lambda):
                print(f"{lamb=}")
                lamb_dict = {
                    'step': integration_path.step_name,
                    'lamb': lamb,
                }
                
                hti_lammps_input_dict =  (
                    (in_json_dict 
                    | lamb_dict
                    | {'conf_file': 'conf.lmp','model': 'graph.pb'}))
                # lmp_str = hti._gen_lammps_input(**hti_lammps_input.model_dump())
                
                lmp_str = self.hti_lammps_adapter.generate_lmp_str(input_dict=hti_lammps_input_dict)

                subtask_name = f"{integration_path.subtasks_dirname}/task.{idx:06d}"
                self._write_subtasks_files(
                    subtasks_dirname=subtasks_dirname,
                    subtask_name=subtask_name,
                    lmp_str=lmp_str,
                    lamb=lamb
                )
        return None
    def _upload_path_files(self, io_handler) -> None:
        """Upload configuration and model files to the integration path directory"""
        io_handler.upload_file(
            file_path='conf.lmp',
            base_dir=os.path.join(self.io_handler.flow_running_dir, self.JOB_DIRNAME)
        )
        
        io_handler.upload_file(
            file_path=self.node_settings.model,
            base_dir=os.path.join(self.io_handler.flow_running_dir, self.JOB_DIRNAME),
            new_file_name='graph.pb'
        )
    
    def _write_subtasks_files(self, subtasks_dirname: str, subtask_name: str, lmp_str: str, lamb: float):
        with self.workflow_services.io_handler.subjobdir_context(subjob_dirname=subtask_name) as io:
            print(f"{subtask_name=}, {io=}")
            io.write_pure_file(file_path='in.lammps', file_content=lmp_str)
            io.write_pure_file(file_path='lambda.out', file_content=str(lamb))
            io.upload_files(
                file_paths=['graph.pb', 'conf.lmp'],
                base_dir=os.path.join(
                    self.io_handler.flow_running_dir, self.JOB_DIRNAME, subtasks_dirname
                    )
            )

            # io.upload_file(file_path='conf.lmp',
            #               base_dir=os.path.join(self.io_handler.flow_running_dir, self.JOB_DIRNAME),
            #               new_file_name='conf.lmp')

#%% 


# %%
class HTIAnalysisInput(BaseModel):
    pass

# class HTILammpsInput(BaseModel):
#     pass



#%%



class SolidHTI(HTILammpsAdapter):
    """Regular solid HTI simulations"""

    
    path_lj_on_tmpl = HTIIntegrationPathTemplate(
        field_name='lambda_lj_on', 
        step_name='lj_on',
        subtasks_dirname='00.lj_on'
    )
    
    path_deep_on_tmpl = HTIIntegrationPathTemplate(
        field_name='lambda_deep_on',
        step_name='deep_on',
        subtasks_dirname='01.deep_on',
    )

    path_spring_off_tmpl = HTIIntegrationPathTemplate(
        field_name='lambda_spring_off',
        step_name='spring_off',
        subtasks_dirname='02.spring_off'
    )


    
    def generate_lmp_str(self, input_dict: Dict[str, Any]) -> str:
        solid_iammps_input = HTISolidLammpsInput(**input_dict)

        lmp_str = hti._gen_lammps_input(
            **solid_iammps_input.model_dump()
        )
        return lmp_str
    
    def analyze_task(self, job_dir: str, free_energy_type: str, manual_pv: float, manual_pv_err: float) -> Dict[str, Any]:

        r = hti.compute_task(
            job=job_dir,
            free_energy_type=free_energy_type,
            manual_pv=manual_pv,
            manual_pv_err=manual_pv_err
        )
        return r
    
    def generate_integration_path_list(self, switch: str) -> List[HTIIntegrationPath]:
        # switch = self.node_settings.switch
        match switch:
            case 'one-step':
                integration_path_list = [self.path_deep_on_tmpl(seq_list=self.node_settings.lambda_deep_on)]
            case 'two-step':
                integration_path_list = [
                    self.path_deep_on_tmpl(seq_list=self.node_settings.lambda_deep_on),
                    self.path_spring_off_tmpl(seq_list=self.node_settings.lambda_spring_off)
                ]
            case 'three-step':
                if self.node_settings.lambda_lj_on is None:
                    raise ValueError("lambda_lj_on is not set")
                integration_path_list = [
                    self.path_lj_on_tmpl(seq_list=self.node_settings.lambda_lj_on),
                    self.path_deep_on_tmpl(seq_list=self.node_settings.lambda_deep_on),
                    self.path_spring_off_tmpl(seq_list=self.node_settings.lambda_spring_off)
                ]
            case _:
                raise ValueError(f"Error option {switch=}")    
        return integration_path_list



    
    
        # with open(os.path.join(analysis_params.job_path, "conf.lmp")) as f:
        #     sys_data = lmp.to_system_data(f.read().split("\n"))
        # natoms = sum(sys_data["atom_numbs"])
        # return hti.post_tasks(
        #     analysis_params.job_path,
        #     natoms=natoms,
        #     method=analysis_params.inte_method,
        #     scheme=analysis_params.scheme
        # )
class HTIIceLammpsInput(BaseModel):

    pass


class IceHTI(HTILammpsAdapter):
    """Ice HTI simulations"""

    path_lj_on_tmpl = HTIIntegrationPathTemplate(
        field_name='lambda_lj_on', 
        step_name='lj_on',
        subtasks_dirname='00.lj_on'
    )
    
    path_deep_on_tmpl = HTIIntegrationPathTemplate(
        field_name='lambda_deep_on',
        step_name='deep_on',
        subtasks_dirname='01.deep_on',
    )

    path_spring_off_tmpl = HTIIntegrationPathTemplate(
        field_name='lambda_spring_off',
        step_name='spring_off',
        subtasks_dirname='02.spring_off'
    )

    def generate_lmp_str(self, input_dict: Dict[str, Any]) -> str:
        ice_iammps_input = HTISolidLammpsInput(**input_dict)

        lmp_str = hti._gen_lammps_input(
            **ice_iammps_input.model_dump()
        )
        return lmp_str
    
    def analyze_task(self, job_dir: str, free_energy_type: str, manual_pv: float, manual_pv_err: float) -> Dict[str, Any]:
        
        args = SimpleNamespace(
            JOB = job_dir,
            type = free_energy_type,
            npt = None,
            pv = manual_pv,
            pv_err = manual_pv_err,
            disorder_corr = self.node_settings.disorder_corr,
            partial_disorder = self.node_settings.partial_disorder,
            inte_method= "inte",
            scheme= "simpson",
            shift = 0.0,
        )

        r = hti_ice.handle_compute(args=args)
        return r
    
    def generate_integration_path_list(self, switch: str) -> List[HTIIntegrationPath]:
        # switch = self.node_settings.switch
        match switch:
            case 'one-step':
                integration_path_list = [self.path_deep_on_tmpl(seq_list=self.node_settings.lambda_deep_on)]
            case 'two-step':
                integration_path_list = [
                    self.path_deep_on_tmpl(seq_list=self.node_settings.lambda_deep_on),
                    self.path_spring_off_tmpl(seq_list=self.node_settings.lambda_spring_off)
                ]
            case 'three-step':
                if self.node_settings.lambda_lj_on is None:
                    raise ValueError("lambda_lj_on is not set")
                integration_path_list = [
                    self.path_lj_on_tmpl(seq_list=self.node_settings.lambda_lj_on),
                    self.path_deep_on_tmpl(seq_list=self.node_settings.lambda_deep_on),
                    self.path_spring_off_tmpl(seq_list=self.node_settings.lambda_spring_off)
                ]
            case _:
                raise ValueError(f"Error option {switch=}")    
        return integration_path_list


class HTIWaterLammpsInput(BaseModel):
    step: str = Field(..., description="HTI step", pattern="^(lj_on|deep_on|spring_off)$")
    conf_file: str = Field(..., validation_alias=AliasChoices('conf_file', 'equi_conf'))
    mass_map: List[float] = Field(..., validation_alias=AliasChoices('mass_map', 'model_mass_map'))
    lamb: float # process control current lambda value
    model: str
    bparam: dict = Field(..., validation_alias=AliasChoices('bparam','bond_param'))
    sparam: dict = Field(..., validation_alias=AliasChoices('sparam','soft_param'))
    nsteps: int
    dt: float = Field(..., validation_alias=AliasChoices('dt','timestep'))
    ens: str # nvt  npt  or npt-iso or nve
    temp: float
    pres: float 
    tau_t: float # not pass in but used
    tau_p: float # not pass in but used
    prt_freq: int = Field(..., validation_alias=AliasChoices('prt_freq','thermo_freq'))
    dump_freq: int
    copies: Optional[List[int]] = Field(default=None, validation_alias=AliasChoices('copies','ncopies')) # deprecated

class WaterHTI(HTILammpsAdapter):
    """Liquid water molecule HTI simulations"""

    path_lj_on_tmpl = HTIIntegrationPathTemplate(
        field_name='lambda_lj_on', 
        step_name='lj_on',
        subtasks_dirname='00.angle_on/'
    )
    
    path_deep_on_tmpl = HTIIntegrationPathTemplate(
        field_name='lambda_deep_on',
        step_name='deep_on',
        subtasks_dirname='01.deep_on',
    )

    path_spring_off_tmpl = HTIIntegrationPathTemplate(
        field_name='lambda_spring_off',
        step_name='spring_off',
        subtasks_dirname='02.spring_off'
    )

    def generate_lmp_str(self, input_dict: Dict[str, Any]) -> str:
        water_iammps_input = HTIWaterLammpsInput(**input_dict)

        lmp_str = hti_water._gen_lammps_input(
            **water_iammps_input.model_dump()
        )
        return lmp_str
    
    def analyze_task(self, job_dir: str, free_energy_type: str, manual_pv: float, manual_pv_err: float) -> Dict[str, Any]:
        
        args = SimpleNamespace(
            JOB = job_dir,
            type = free_energy_type,
            npt = None,
            pv = manual_pv,
            pv_err = manual_pv_err,
            disorder_corr = self.node_settings.disorder_corr,
            partial_disorder = self.node_settings.partial_disorder,
            inte_method= "inte",
            scheme= "simpson",
            shift = 0.0,
        )

        r = hti_water.handle_compute(args=args)
        return r
    
    def generate_integration_path_list(self, switch: str) -> List[HTIIntegrationPath]:
        # switch = self.node_settings.switch
        match switch:
            case 'one-step':
                integration_path_list = [self.path_deep_on_tmpl(seq_list=self.node_settings.lambda_deep_on)]
            case 'two-step':
                integration_path_list = [
                    self.path_deep_on_tmpl(seq_list=self.node_settings.lambda_deep_on),
                    self.path_spring_off_tmpl(seq_list=self.node_settings.lambda_spring_off)
                ]
            case 'three-step':
                if self.node_settings.lambda_lj_on is None:
                    raise ValueError("lambda_lj_on is not set")
                integration_path_list = [
                    self.path_lj_on_tmpl(seq_list=self.node_settings.lambda_lj_on),
                    self.path_deep_on_tmpl(seq_list=self.node_settings.lambda_deep_on),
                    self.path_spring_off_tmpl(seq_list=self.node_settings.lambda_spring_off)
                ]
            case _:
                raise ValueError(f"Error option {switch=}")    
        return integration_path_list

# class WaterHTI(HTILammpsAdapter):
#     """Water molecule HTI simulations"""
    
#     def generate_task(self, input_dict: ) -> None:
#         with open(input_params.param_file, 'r') as f:
#             jdata = json.load(f)
#         hti_water.make_tasks(input_params.output, jdata)
    
#     def analyze_task(self, analysis_params: HTIAnalysisInput) -> Dict[str, Any]:
#         with open(os.path.join(analysis_params.job_path, "conf.lmp")) as f:
#             sys_data = lmp.to_system_data(f.read().split("\n"))
#         natoms = sum(sys_data["atom_numbs"])
#         nmols = natoms // 3
#         return hti_water.post_tasks(
#             analysis_params.job_path,
#             nmols,
#             method=analysis_params.inte_method
#         )


class HTIIdealLiquidInput(BaseModel, extra='ignore'):
    step: str = Field(..., description="Ideal HTI", pattern="^(soft_on|deep_on|soft_off)$")
    conf_file: str = Field(..., validation_alias=AliasChoices('conf_file', 'equi_conf'))
    mass_map: List[float] = Field(..., validation_alias=AliasChoices('mass_map', 'model_mass_map'))
    lamb: float # process control current lambda value
    soft_param: dict = Field(..., validation_alias=AliasChoices('soft_param','sparam'))
    model: str
    nsteps: int
    timestep: float = Field(..., validation_alias=AliasChoices('timestep','dt'))

    ens: str # must be `nvt` or `nvt-langevin` controller by langevin
    pres: float # pass in but will not be used
    temp: float
    tau_t: float # not pass in but used
    tau_p: float # not pass in but used

    thermo_freq: int = Field(..., validation_alias=AliasChoices('thermo_freq','stat_freq'))
    dump_freq: int
    copies: Optional[List[int]] = Field(default=None, validation_alias=AliasChoices('copies','ncopies')) # deprecated
    # norm_style: str = Field(default='first')
    if_meam: bool
    meam_model: dict|None
    pass


class LiquidHTI(HTILammpsAdapter):
    """Liquid HTI simulations"""


    path_soft_on_tmpl = HTIIntegrationPathTemplate(
        field_name='lambda_soft_on', 
        step_name='soft_on',
        subtasks_dirname='00.soft_on'
    )

    path_deep_on_tmpl = HTIIntegrationPathTemplate(
        field_name='lambda_deep_on',
        step_name='deep_on',
        subtasks_dirname='01.deep_on',
    )

    path_soft_off_tmpl = HTIIntegrationPathTemplate(
        field_name='lambda_soft_off',
        step_name='soft_off',
        subtasks_dirname='02.soft_off'
    )

    
    def generate_lmp_str(self, input_dict: Dict[str, Any]) -> str:

        ideal_iammps_input = HTIIdealLiquidInput(**input_dict)
        lmp_str = hti_liq._gen_lammps_input_ideal(
            **ideal_iammps_input.model_dump()
        )
        return lmp_str

    
    def analyze_task(self, job_dir: str, free_energy_type: str, manual_pv: float, manual_pv_err: float) -> Dict[str, Any]:
        # with open(os.path.join(analysis_params.job_path, "conf.lmp")) as f:
        #     sys_data = lmp.to_system_data(f.read().split("\n"))
        # natoms = sum(sys_data["atom_numbs"])
        r = hti_liq.compute_task(
            job=job_dir, free_energy_type=free_energy_type, manual_pv=manual_pv, manual_pv_err=manual_pv_err
        )
        return r
    

    def generate_integration_path_list(self, switch: str) -> List[HTIIntegrationPath]:
        match switch :
            case 'one-step':
                raise ValueError("one-step is not supported for liquid HTI")
            case 'two-step':
                raise ValueError("two-step is not supported for liquid HTI")
            case 'three-step':
                if self.node_settings.lambda_lj_on is None:
                    raise ValueError("lambda_lj_on is not set")
                integration_path_list = [
                    self.path_soft_on_tmpl(seq_list=self.node_settings.lambda_lj_on), # note lambda_lj_on is used for soft_on for compatibility Solid reason
                    self.path_deep_on_tmpl(seq_list=self.node_settings.lambda_deep_on),
                    self.path_soft_off_tmpl(seq_list=self.node_settings.lambda_spring_off) # note lambda_lj_off is used for soft_off for compatibility Solid reason
                ]
            case _:
                raise ValueError(f"Error option {switch=}")
        return integration_path_list
        # return hti_liq.post_tasks(analysis_params.job_path, natoms)

# class IceHTI(HTILammpsAdapter):
#     """Ice HTI simulations"""
    
#     def generate_task(self, input_params: HTILammpsInput) -> None:
#         with open(input_params.param_file, 'r') as f:
#             jdata = json.load(f)
#         crystal_type = "frenkel" if jdata.get("crystal") == "frenkel" else "vega"
#         hti.make_tasks(
#             input_params.output,
#             jdata,
#             ref="einstein",
#             switch=input_params.switch
#         )
    
#     def analyze_task(self, analysis_params: HTIAnalysisInput) -> Dict[str, Any]:
#         return hti_ice.handle_compute(analysis_params)

#%%
# step1_data = np.loadtxt('../examples/Sn_beta_quicktest/beta_5GPa_200K/HTI_sim/new_job/00.lj_on/hti.out')
# step2_data = np.loadtxt('../examples/Sn_beta_quicktest/beta_5GPa_200K/HTI_sim/new_job/01.deep_on/hti.out')
# step3_data = np.loadtxt('../examples/Sn_beta_quicktest/beta_5GPa_200K/HTI_sim/new_job/02.spring_off/hti.out')

# %%

# lmbda = step1_data[:, 0]
# U_integrand = step1_data[:, 1]
# all_err = step1_data[:, 2] # U_integrand's error
# Ud = step1_data[:, 3] # all_ed / all_lambda
# Us = step1_data[:, 4] # all_es / (1 - all_lambda)
# Ud_err = step1_data[:, 5] # all_ed_err / all_lambda
# Us_err = step1_data[:, 6] # all_es_err / (1 - all_lambda)
# etot = step1_data[:, 7] # all_etot / natoms
# spring_eng = step1_data[:, 8] # all_es
# enthalpy = step1_data[:, 9] # all_enthalpy
# msd_xyz = step1_data[:, 10] # all_msd_xyz, indicates liquid or solig structure

# %%

def plot_hti_results(folder_list: List[str], base_path: str) -> str:
    """Plot HTI simulation results with statistical errors in 2x2 subplots
    
    Args:
        folder_list: List of folder names containing hti.out files (e.g., ['00.lj_on', '01.deep_on'])
        base_path: Base path to the HTI simulation folders
    
    Returns:
        Base64 encoded string of the plot image
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    plt.rcParams.update({'font.size': 16})
    axes = axes.ravel()
    fig.suptitle(f'HTI Results - {base_path}', fontsize=16, y=0.98)
    
    # Plot individual steps
    for idx, folder in enumerate(folder_list):
        target_hti_out_path = os.path.join(base_path, folder, 'hti.out')
        data = np.loadtxt(target_hti_out_path)
        lmbda = data[:, 0]
        U_integrand = data[:, 1]
        U_integrand_err = data[:, 2]
        diff_e, stat_err, sys_integ_err = integrate_range_hti(lmbda, U_integrand, U_integrand_err, scheme='simpson')

        axes[idx].errorbar(lmbda, U_integrand, yerr=U_integrand_err, 
                          fmt='o-', label=folder,
                          color='blue', capsize=5, capthick=2,
                          ecolor='red', alpha=0.7, elinewidth=1)
        
        title = (f'HTI Integrand vs λ - {folder}\n'
                f'ΔU = {diff_e:.6e} ± ({stat_err:.3e}, {sys_integ_err:.3e}) eV')
        
        axes[idx].set_title(title, pad=10)
        axes[idx].set_xlabel('λ')
        axes[idx].set_ylabel('U Integrand (eV)')
        axes[idx].grid(True, linestyle='--', alpha=0.7)
        axes[idx].legend()
        
        # Adjust y-axis range
        y_range = np.max(U_integrand) - np.min(U_integrand)
        axes[idx].set_ylim(np.min(U_integrand) - y_range*0.1, 
                          np.max(U_integrand) + y_range*0.1)
    
    # Remove empty subplot if less than 4 datasets
    for idx in range(len(folder_list), 4):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    
    # Save figure
    fig.savefig('hti_results.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # Convert to base64 string
    with BytesIO() as buf:
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return img_base64


#%%

fig_md_base64_tmpl = """### HTI Result Plot
![HTI Result](data:image/png;base64,{img_base64})
"""

def get_hti_result_fig_base64_md(folder_list: List[str], base_path: str) -> str:



    """Generate markdown string containing the HTI result plot"""
    img_base64 = plot_hti_results(folder_list=folder_list, 
                                base_path=base_path)
    fig_base64_md = fig_md_base64_tmpl.format(img_base64=img_base64)
    return fig_base64_md

#%%


# Generate plot and markdown
# fig_md_str = get_hti_result_str(folder_list=['00.lj_on', '01.deep_on', '02.spring_off'], 
#                             base_path='../examples/Sn_beta_quicktest/beta_5GPa_200K/HTI_sim/new_job')

#%%
# Save markdown file
# with open('hti_result.md', 'w') as f:
#     f.write(fig_md_str)


# print(lj_on_data)









#%%

