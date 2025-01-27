import os
import json
from typing import List, Dict, TypeVar, TypedDict, Union, Any, Type, NamedTuple, Optional
from  pydantic import BaseModel, Field, AliasChoices
from .base import SimulationBase, CreateFromTemplateMixin, FreeEnergyValuePoint, transfer_matching_fields
from ...lib.utils import parse_seq
from ... import hti


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
    SimulationBase[HTISimulationNodedata,  # NodedataType,
                   Union[BaseModel, Dict[str, Any], NamedTuple],  #  InitializationType
                   HTIResultData] # ReturnType
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
                io.upload_files(file_paths=["out.lmp"], base_dir=os.path.join(self.io_handler.flow_running_dir, self.JOB_DIRNAME))
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
                base_dir=os.path.join(self.io_handler.flow_running_dir, self.JOB_DIRNAME)
            )

