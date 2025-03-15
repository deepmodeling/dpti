# %%
import asyncio
import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
)

# %%
import matplotlib.pyplot as plt
import numpy as np
from prefect.artifacts import create_markdown_artifact
from pydantic import AliasChoices, BaseModel, Field

from dpti.workflows.simulations.base import SimulationBase, transfer_matching_fields, FreeEnergyValuePoint, SettingsBase, FlowRunInfo
from dpti.lib.utils import parse_seq
from dpti import ti, ti_water
from dpti.workflows.service.file_handler import get_current_io
from dpti.workflows.simulations.result_base import ResultDataBase, ArtifactInfo
from dpti.lib.lammps import get_natoms
from dpti.workflows.service.workflow_service_module import BasicWorkflowServices

from dpti import ti
from dpti.lib.utils import parse_seq
from dpti.workflows.service.file_handler import get_current_io

# %%
from .base import (
    FreeEnergyValuePoint,
    SettingsBase,
    SimulationBase,
    transfer_matching_fields,
)
from .result_base import ArtifactInfo, ResultDataBase

# from


class TISimulationSettings(SettingsBase, extra="allow"):
    # flow_trigger_dir: Optional[str] = Field(default=None)
    # flow_running_dirname: Optional[str] = Field(default=None)
    # flow_run_info: FlowRunInfo
    # conf_file: str = Field(..., validation_alias=AliasChoices('conf_file', 'equi_conf'))
    equi_conf: str = Field(..., validation_alias=AliasChoices("conf_file", "equi_conf"))
    ncopies: List[int] = Field(
        default=[1, 1, 1], validation_alias=AliasChoices("ncopies", "copies")
    )
    model: str
    mass_map: List[float] = Field(
        ..., validation_alias=AliasChoices("mass_map", "model_mass_map")
    )
    nsteps: int
    timestep: float
    ens: str
    path: str = Field(..., validation_alias=AliasChoices("path", "ti_path"))
    temp_seq: Optional[List[str]] = None
    pres_seq: Optional[List[str]] = None
    temp: Optional[float] = None
    pres: Optional[float] = None
    tau_t: float
    tau_p: float
    thermo_freq: int
    dump_freq: int = 10000
    stat_skip: int
    stat_bsize: int
    if_water: bool
    if_meam: bool
    meam_model: Optional[Dict[str, Any]] = None

    # TEMPLATE_DEFAULT_JSON:ClassVar[str] = "examples/ti.t.json"


# TITemplateMixin:Type = CreateFromTemplateMixin.configure(
#     mixin_cls_name="TITemplateMixin",
#     TEMPLATE_DEFAULT_JSON='ti.t.json',
#     TEMPLATE_ADDITIONAL_REQUIRED_FIELDS={
#         # 'temp': float,
#         # 'pres': float,
#         # 'ens': str
#     }
# )


class TISimulationUpstreamData(BaseModel):
    free_energy_ref_value_point: FreeEnergyValuePoint


class TISimulationResultData(ResultDataBase):
    all_temps: List[float] = Field(..., description="Temperature points")
    all_press: List[float] = Field(..., description="Pressure points")
    all_fe: List[float] = Field(..., description="Free energy values")
    all_fe_stat_err: List[float] = Field(..., description="Statistical errors")
    all_fe_inte_err: List[float] = Field(..., description="Integration errors")
    all_fe_tot_err: float = Field(..., description="Total error")
    # pass
    # ti_result_data: TiResultData


class TISimulationNodeData(
    TISimulationSettings,
    # TILammpsInput,
    # TITemplateMixin
):
    node_settings: TISimulationSettings
    node_upstream_data: TISimulationUpstreamData
    node_result_data: TISimulationResultData
    # node
    # free_energy_ref_value_point: FreeEnergyValuePoint


class ThermodynamicPoint(BaseModel):
    """Single thermodynamic state point data."""

    p: float = Field(..., description="Pressure")
    p_err: float = Field(..., description="Pressure error")
    v: float = Field(..., description="Volume")
    v_err: float = Field(..., description="Volume error")
    e: float = Field(..., description="Energy")
    e_err: float = Field(..., description="Energy error")
    h: float = Field(..., description="Enthalpy")
    h_err: float = Field(..., description="Enthalpy error")
    t: float = Field(..., description="Temperature")
    t_err: float = Field(..., description="Temperature error")
    pv: float = Field(..., description="PV term")
    pv_err: float = Field(..., description="PV term error")


class TISimulationFreeEnergyData(ResultDataBase):
    """Result data for TI simulation."""

    all_temps: List[float] = Field(..., description="Temperature points")
    all_press: List[float] = Field(..., description="Pressure points")
    all_fe: List[float] = Field(..., description="Free energy values")
    all_fe_stat_err: List[float] = Field(
        ..., description="Statistical errors in free energy"
    )
    all_fe_inte_err: List[float] = Field(
        ..., description="Integration errors in free energy"
    )
    all_fe_tot_err: float = Field(..., description="Total error in free energy")


class TISimulationResult(ResultDataBase):
    """Thermodynamic simulation result data."""

    start_point_info: ThermodynamicPoint = Field(
        ..., description="Initial state point data"
    )
    end_point_info: ThermodynamicPoint = Field(
        ..., description="Final state point data"
    )
    ti_free_energy_data: TISimulationFreeEnergyData = Field(
        ..., description="Thermodynamic data across all states"
    )
    # result_data: TISimulationResultData = Field(
    #     default_factory=TISimulationResultData,
    #     description="Raw simulation data like temperatures and pressures"
    # )


class TISimulation(
    SimulationBase[
        TISimulationNodeData,  # NodeDataType,
        TISimulationSettings,  # SettingsType
        TISimulationResultData,
    ]  # ReturnType
):
    # DEFAULT_NODEDATA_JSON = "ti.t.json"
    # JOB_DIRNAME = "TI_t_sim/new_job/"
    JOB_DIRNAME = "TI_sim/new_job/"
    UPLOAD_LOCAL_FILES = []
    UPLOAD_LOCAL_FILES_FIELDS = ["model", "equi_conf"]
    settings_data_type: Type[TISimulationSettings] = TISimulationSettings
    NODEDATA_FILENAME = "ti_settings.json"
    # NODEDATA_FILENAME = "ti_settings.json"
    # node_settings: TISimulationSettings
    nodedata_type: Type[TISimulationNodeData] = TISimulationNodeData
    workflow_services: BasicWorkflowServices
    # def __init__(self, updates={}, template_json:Optional[str]=None):
    #     self.updates = updates
    #     self.template_json = template_json
    #     self.updated_nodedata = self.nodedata_type.from_template(
    #         updates=updates, template_json=template_json)

    # @task
    def _prepare(self) -> Dict[str, Any]:
        with self.workflow_services.io_handler.jobdir_context(
            job_dirname=self.JOB_DIRNAME
        ) as io_handler:
            self.upload_predefined_files(
                upload_local_files_fields=self.UPLOAD_LOCAL_FILES_FIELDS
            )
            path = self.node_settings.path

            ti_integration_path = TIIntegraionPath.get_instance(path=path)
            self.ti_integration_path = ti_integration_path

            thermo_path_seq = getattr(
                self.node_settings, self.ti_integration_path.path_field_name
            )
            print(f" {self.ti_integration_path=} {thermo_path_seq=}")

            thermo_points_list = ti_integration_path.parse_seq_list(
                thermo_path_seq=thermo_path_seq
            )

            const_thermo_name = ti_integration_path.const_thermo_name
            extra_thermo_info_dict = {
                const_thermo_name: getattr(self.node_settings, const_thermo_name),
            }
            print(f"{extra_thermo_info_dict=}")

            # in_json_dict = (self.updated_nodedata.model_dump() | extra_thermo_info_dict)
            in_json_dict = (
                self.node_settings.model_dump()
                | extra_thermo_info_dict
                | {"prev_results": self.prev_results}
            )

            self.io_handler.write_pure_file(
                file_path=self.NODEDATA_FILENAME,
                file_content=json.dumps(obj=in_json_dict, indent=4),
            )
            # thermo_path =
            # task_dir = os.path.join(job_abs_dir, "task.%06d" % ii)

            for idx, thermo_point in enumerate(thermo_points_list):
                subtask_name = f"task.{idx:06d}/"
                with self.workflow_services.io_handler.subjobdir_context(
                    subjob_dirname=subtask_name
                ) as io:
                    # print(f"{subtask_name=}, {io=}")

                    lammps_input_dict = transfer_matching_fields(
                        from_obj=self.node_settings, to_type=TILammpsInput
                    )

                    print(f"{lammps_input_dict=}")
                    thermo_point_dict = {
                        ti_integration_path.point_field_name: thermo_point
                    }
                    lmp_str = ti._gen_lammps_input(
                        **(
                            lammps_input_dict
                            | extra_thermo_info_dict
                            | thermo_point_dict
                        )
                    )
                    io.write_pure_file(file_path="in.lammps", file_content=lmp_str)
                    io.write_pure_file(
                        file_path="thermo.out", file_content=str(thermo_point)
                    )
                    equi_conf = self.node_settings.equi_conf
                    print(f"{equi_conf=}")
                    io.upload_files(
                        file_paths=[
                            "graph.pb",
                            equi_conf,
                        ],
                        base_dir=io.flow_trigger_dir,
                    )
            return {}
            # task_dir = os.path.join(, "task.%06d" % ii)
            # task_abs_dir = create_path(task_dir)

    # @task
    def _run(self) -> str:
        print(f"TISimulation instance to submit {self.job_dir=}")
        loop = asyncio.get_event_loop()
        try:
            submission_hash = loop.run_until_complete(
                self.workflow_services.job_executor.group_submit(
                    job_dir=self.job_dir,
                    subtasks_template="./task*",
                    command="ln -s ../graph.pb ./; lmp -i in.lammps",
                )
            )
            print(f"_run: submission ends with {submission_hash=}")
        except Exception as e:
            print(f"Error during job execution: {e=}")
            raise e
        return submission_hash

    # @task
    def _extract(self) -> TISimulationResultData:
        # raise RuntimeError
        # self.io_handler.use_job_info(job_dirname=self.JOB_DIRNAME)
        # Eo = self.updated_nodedata.hti_to_ti_result['']

        with self.workflow_services.io_handler.jobdir_context(
            job_dirname=self.JOB_DIRNAME
        ) as io_handler:
            # info = hti.post_tasks(io_handler.job_dir)
            Eo = self.prev_results["free_energy_ref_value_point"]["gibbs_free_energy"]
            Eo_err = self.prev_results["free_energy_ref_value_point"][
                "gibbs_free_energy_err"
            ]
            if self.node_settings.path == "t":
                To = self.node_settings.temp
            elif self.node_settings.path == "p":
                To = self.node_settings.pres
            else:
                raise RuntimeError(
                    f"Known To value {self.node_settings.path=} {self.node_settings=}"
                )
            if not self.node_settings.if_water:
                ti_info = ti.compute_task(
                    job=self.job_dir,
                    inte_method="inte",
                    Eo=Eo,  # free energy value of given reference thermo condition point.
                    Eo_err=Eo_err,
                    To=To,  # the known free energy value's thermo conditional.
                    # natoms=
                )
            else:
                equi_conf = self.node_settings.equi_conf
                equi_conf_abs_path = os.path.join(self.job_dir, equi_conf)
                natoms = get_natoms(equi_conf_abs_path)
                water_nmols = natoms // 3
                ti_info = ti.post_tasks(
                    iter_name=self.job_dir,
                    jdata=self.node_settings.model_dump(),
                    Eo=Eo,
                    Eo_err=Eo_err,
                    To=To,
                    natoms=water_nmols,
                    scheme="simpson",
                    shift=0.0,
                )  # the known free energy value's thermo conditional.
            # extract_result['data']['all_temps'] = [float(t) for t in extract_result['data']['all_temps']]
            print(f"{ti_info=}")

            ti_summary_md = get_ti_summary_md(result_info=ti_info)
            io_handler.write_pure_file(
                file_path="ti_summary.md", file_content=ti_summary_md
            )

        with self.workflow_services.io_handler.jobdir_context(
            job_dirname=self.JOB_DIRNAME
        ) as io_handler:
            ti_simulation_result = TISimulationResult(
                start_point_info=ThermodynamicPoint(
                    **ti_info["start_point_info"],
                ),
                end_point_info=ThermodynamicPoint(
                    **ti_info["end_point_info"],
                ),
                ti_free_energy_data=TISimulationFreeEnergyData(
                    **ti_info["data"],
                ),
                artifacts=[
                    ArtifactInfo(
                        path=Path("ti_summary.md"),
                        description="TI Simulation Report",
                        mime_type="text/markdown",
                    )
                ],
            )
            io_handler.write_pure_file(
                file_path="ti_simulation_result.json",
                file_content=ti_simulation_result.model_dump_json(),
            )

        create_markdown_artifact(
            key="ti-report",
            markdown=ti_summary_md,
            description="TI Simulation Report",
        )

        # ti_result_data = TISimulationResultData(
        #     **ti_info['data'],
        # )
        # result_file_path = os.path.join(io_handler.job_dir, "result.json")
        # return ti_result_data
        return ti_info


# %%


# %%


# %%


# if ti_path == "t":
# with open(os.path.join(work_base_abs_dir, "ti.t.json")) as j:
#     ti_jdata = json.load(j)
#     task_jdata = ti_jdata.copy()
#     task_jdata["pres"] = start_info["target_pres"]
#     job_dir = "TI_t_sim"
# ti_path = start_info["ti_path"]


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


# %%


TIIntegrationPathType = TypeVar("TIIntegrationPathType", bound="TIIntegraionPath")


class TIIntegraionPath:
    _instances = {}

    def __init__(
        self,
        path: str,
        point_field_name: str,
        path_field_name: str,
        const_thermo_name: str,
        job_dirname: str,
    ):
        self.path = path  # 't' or 'p'
        self.point_field_name = point_field_name
        self.path_field_name = path_field_name
        self.const_thermo_name = const_thermo_name
        self.job_dirname = job_dirname
        self.is_parsed = False
        self.thermo_seq = []
        TIIntegraionPath._instances[path] = self

    def __repr__(self):
        r = (
            f"working for {self.path=}"
            f" {self.path_field_name=}"
            f" {self.job_dirname=}"
            f" {self.thermo_seq=}"
        )
        return r

    @classmethod
    def get_instance(
        cls: Type[TIIntegrationPathType], path: str
    ) -> TIIntegrationPathType:
        instance = cls._instances[path]
        return instance

    def parse_seq_list(self, thermo_path_seq):
        self.thermo_path_seq = thermo_path_seq
        self.thermo_points_list = list(
            parse_seq(self.thermo_path_seq, protect_eps=None)
        )
        self.is_parsed = True
        return self.thermo_points_list
        # temp_list = parse_seq(temp_seq)


t_ti_path = TIIntegraionPath(
    path="t",
    point_field_name="temp",
    path_field_name="temp_seq",
    const_thermo_name="pres",
    job_dirname="TI_t_sim",
)
p_ti_path = TIIntegraionPath(
    path="p",
    point_field_name="pres",
    path_field_name="pres_seq",
    const_thermo_name="temp",
    job_dirname="TI_p_sim",
)


class TILammpsInput(BaseModel):
    # equi_conf: str = Field(..., validation_alias=AliasChoices('conf_file', 'equi_conf'))
    conf_file: str = Field(..., validation_alias=AliasChoices("conf_file", "equi_conf"))
    mass_map: List[float] = Field(
        ..., validation_alias=AliasChoices("mass_map", "model_mass_map")
    )
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
    copies: List[int] = Field(
        [1, 1, 1], validation_alias=AliasChoices("copies", "ncopies")
    )
    if_meam: bool
    meam_model: Optional[Dict[str, Any]] = None
    # lamb: float # process control current lambda value
    # step: str # process control
    # m_spring_k: List[float]
    # ens: str # must be `nvt` or `nvt-langevin` controller by langevin
    # pres: float # pass in but will not be used
    # tau_t: float # not pass in but used
    # tau_p: float # not pass in but used


# %%
# a = TISimulationNodeData.from_template(updates={})


# print(a)
def get_ti_summary_md(result_info: Dict[str, Any]) -> str:
    header = """T(ctrl)          P(ctrl)                     F   stat_err   inte_err    tot_err"""
    rows = []
    all_temps = result_info["data"]["all_temps"]
    all_press = result_info["data"]["all_press"]
    all_fe = result_info["data"]["all_fe"]
    all_fe_stat_err = result_info["data"]["all_fe_stat_err"]
    all_fe_inte_err = result_info["data"]["all_fe_inte_err"]
    all_fe_tot_err = np.linalg.norm([all_fe_stat_err, all_fe_inte_err], axis=0).tolist()

    for temp, press, fe, fe_stat_err, fe_inte_err, fe_tot_err in zip(
        all_temps, all_press, all_fe, all_fe_stat_err, all_fe_inte_err, all_fe_tot_err
    ):
        rows.append(
            f"{temp:9.2f}  {press:15.8e}  {fe:20.12f}  {fe_stat_err:9.2e}  {fe_inte_err:9.2e}  {fe_tot_err:9.2e}"
        )

    result_fig_base64_md_str = get_ti_result_fig_base64_md_str(result_info=result_info)

    extra_info = f"extra info:\n {result_info}"
    md = (
        header
        + "\n"
        + "\n".join(rows)
        + "\n"
        + result_fig_base64_md_str
        + "\n"
        + extra_info
    )
    return md


def plot_ti_result(result_info: dict) -> str:
    """Plot TI simulation results with both statistical and total errors.

    Args:
        result_info: Dictionary containing TI simulation results
    """
    data = result_info["data"]
    temps = np.array(data["all_temps"])
    fe = np.array(data["all_fe"])
    fe_stat_err = np.array(data["all_fe_stat_err"])
    fe_inte_err = np.array(data["all_fe_inte_err"])
    fe_tot_err = np.sqrt(fe_stat_err**2 + fe_inte_err**2)  # total error
    # print(f"{temps=} {fe=} {fe_stat_err=} {fe_inte_err=} {fe_tot_err=}")
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({"font.size": 14})

    # Plot total error first (will be in the background)
    plt.errorbar(
        temps,
        fe,
        yerr=fe_tot_err,
        fmt="o-",
        label="Free energy total error",
        color="green",
        capsize=5,
        capthick=2,
        ecolor="red",
        elinewidth=2,
    )

    # Plot statistical error second (will be in the foreground)
    plt.errorbar(
        temps,
        fe,
        yerr=fe_inte_err,
        fmt="o-",
        label="integration error",
        color="blue",
        capsize=2,
        capthick=1,
        ecolor="green",
        elinewidth=1,
    )

    for i, (t, f, tot_err) in enumerate(zip(temps, fe, fe_tot_err)):
        plt.annotate(
            f"{f:.6f}eV,{t:.0f}K\n±{tot_err*1000:.2f}meV (per atom)",  # keep 3 decimal places
            xy=(t, f),  # the point to annotate
            xytext=(0, -30) if (i % 2) else (0, 5),  # distance to move the text
            textcoords="offset points",  # use offset to position the text
            ha="left",  # horizontal alignment
            va="top" if (i % 2) else "bottom",  # vertical alignment
            fontsize=6,
        )

    plt.xlabel("Temperature (K)")
    plt.ylabel("Free Energy (eV)")
    fig_title_tmpl = (
        "Free Energy vs Temperature with Error Analysis.\n job_dir={job_dir}"
    )
    try:
        current_io = get_current_io()
    except NameError:
        current_io = None

    if current_io:
        job_dir = current_io.job_dir
        parts = Path(job_dir).parts
        job_dir_short = "/".join(parts[-6:])
        fig_title = fig_title_tmpl.format(job_dir=job_dir_short)
    else:
        fig_title = fig_title_tmpl.format(job_dir="unknown")
    print(f"plot_ti_result: {fig_title=}")
    plt.title(fig_title)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    y_range = np.max(fe) - np.min(fe)
    plt.ylim(np.min(fe) - y_range * 0.1, np.max(fe) + y_range * 0.1)
    plt.tight_layout()

    # Save figure
    fig = plt.gcf()
    plt.show(block=False) if not current_io else None
    # fig.savefig('ti_result.png', dpi=300, bbox_inches='tight')

    with BytesIO() as buf:
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    return img_base64


ti_result_fig_base64_md_str = """### TI Result Plot
![TI Result](data:image/png;base64,{img_base64})
"""


def get_ti_result_fig_base64_md_str(result_info: dict) -> str:
    img_base64 = plot_ti_result(result_info=result_info)
    md_str = ti_result_fig_base64_md_str.format(img_base64=img_base64)
    return md_str


# %%


# with open('../examples/Sn_beta_quicktest/TI_beta_path_t_200K_50000bar_run1/TI_sim/new_job/result.json', 'r') as f:
#     solid_result_info = json.load(f)

# with open('../examples/Sn_beta_quicktest/TI_liquid128_path_t_800K_50000bar_run2/TI_sim/new_job/result.json', 'r') as f:
#     liquid_result_info = json.load(f)

# solid_img_base64 = plot_ti_result(result_info=solid_result_info)
# liquid_img_base64 = plot_ti_result(result_info=liquid_result_info)

# %%


# %%

# %%

# comparison_img_base64 = plot_free_energy_comparison(
#     phase1_result=solid_result_info,
#     phase2_result=liquid_result_info,
#     phase1_label="Solid phase",
#     phase2_label="Liquid phase"
# )


# %%
