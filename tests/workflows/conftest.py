import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import dpti
from dpti.workflows.service.workflow_service_module import WorkflowServiceModule
from dpti.workflows.service.file_handler import IOHandler
from dpti.workflows.simulations.equi_sim import NPTEquiSimulationSettings
from dpti.workflows.simulations.hti_sim import HTISimulationSettings

@pytest.fixture
def workflow_test_dir(tmp_path):
    """创建临时工作目录"""
    return tmp_path / "workflow_test"

@pytest.fixture
def mock_workflow_service():
    """模拟工作流服务"""
    service = Mock(spec=WorkflowServiceModule)
    service.io_handler = Mock(spec=IOHandler)
    service.io_handler.flow_running_dir = "/mock/flow/dir"
    return service

@pytest.fixture
def sample_npt_settings():
    """NPT模拟的示例设置"""
    return NPTEquiSimulationSettings(
        equi_conf="test.lmp",
        model="model.pb",
        mass_map=[1.0, 1.0],
        nsteps=1000,
        timestep=0.002,
        ens="npt",
        temp=300,
        pres=1.0,
        tau_t=0.1,
        tau_p=0.5,
        thermo_freq=100,
        dump_freq=1000,
        stat_skip=100,
        stat_bsize=10,
        if_meam=False
    )

@pytest.fixture
def sample_hti_settings():
    """HTI模拟的示例设置"""
    return HTISimulationSettings(
        equi_conf="test.lmp",
        model="model.pb",
        mass_map=[1.0, 1.0],
        lambda_lj_on=["0.0", "1.0"],
        lambda_deep_on=["0.0", "1.0"],
        lambda_spring_off=["0.0", "1.0"],
        protect_eps=0.1,
        spring_k=1.0,
        soft_param={"alpha": 0.5},
        crystal="fcc",
        nsteps=1000,
        timestep=0.002,
        temp=300,
        pres=1.0,
        if_meam=False
    )

@pytest.fixture
def mock_prefect_context():
    """模拟Prefect上下文"""
    with patch("prefect.context.get_run_context") as mock_context:
        mock_context.return_value.flow_run.parameters = {
            "flow_trigger_dir": "/mock/flow/dir",
            "config_yaml": "test_config.yaml"
        }
        yield mock_context

@pytest.fixture
def sample_workflow_results():
    """示例工作流结果"""
    return {
        "npt": {
            "pv": 1.0,
            "pv_err": 0.1,
            "temperature": 300,
            "pressure": 1.0
        },
        "hti": {
            "free_energy": -10.5,
            "free_energy_err": 0.2,
            "integration_paths": ["lj_on", "deep_on", "spring_off"]
        }
    }

def mock_task(fn):
    """replace @task in testing environment"""
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper

def mock_flow(fn):
    """replace @flow in testing environment"""
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


