from prefect import flow
from pydantic import BaseModel, computed_field, Field
from typing import Optional
from functools import cached_property
import os, yaml
from typing import Union, Dict, Any
from pathlib import Path
# 定义一个简单的Pydantic模型
class SimulationConfig(BaseModel):
    temperature: float
    pressure: float
    simulation_steps: int = 1000

    some_yaml: Union[Path, str, Dict[str, Any], BaseModel] = Field(default={}, 
    description=("support Path /home/user1/some.yaml \n"
        'or str the yaml content like "temperature: 111\nnote: b2b" \n'
        'or a json dict {"temperature":111, "note": "b2b"}'))

    @computed_field
    @cached_property
    def temp_str(self) -> str:
        temp_str = f"{self.temperature}K"
        return temp_str

    @computed_field
    @cached_property
    def yaml_dict(self) -> Dict[str, Any]:
        """解析some_yaml为字典，无论它是何种形式"""
        # 已经是字典
        print(f"to get yaml_dict: {self.some_yaml=} to be parsed")
        yaml_dict = {}
        if isinstance(self.some_yaml, dict):
            yaml_dict = self.some_yaml
            
        # 处理字符串情况
        if isinstance(self.some_yaml, str):
            # 文件路径
            if os.path.exists(self.some_yaml) and self.some_yaml.endswith(('.yaml', '.yml')):
                with open(self.some_yaml, 'r') as f:
                    yaml_dict = yaml.safe_load(f) or {}
            
            # YAML内容字符串
            try:
                parsed = yaml.safe_load(self.some_yaml)
                if isinstance(parsed, dict):
                    yaml_dict = parsed
            except Exception:
                pass  # 解析失败，返回空字典
        
        print(f"parse finished, yaml_dict: {yaml_dict=}")
        # 不可解析情况返回空字典
        return yaml_dict


@flow(log_prints=True)
def free_energy_workflow(thermo_input: SimulationConfig):
    """
    一个简单的工作流，接收SimulationConfig并打印其内容
    """
    print(f"开始模拟计算，参数如下:")
    print(f"温度: {thermo_input.temperature} K")
    print(f"压力: {thermo_input.pressure} bar")
    print(f"模拟步数: {thermo_input.simulation_steps}")
    
    # 在实际应用中，这里会有更多的计算逻辑
    result = thermo_input.temperature * thermo_input.pressure / 100
    print(f"模拟计算结果: {result}")
    print(f"thermo_input: {thermo_input=}")
    print(f"temp_str: {thermo_input.temp_str=}")
    
    return {
        "thermo_input": thermo_input.model_dump(),
        "result": result
    }


if __name__ == "__main__":
    # 部署工作流
    free_energy_workflow.serve(
        name="test-basemodel-deployment",
        tags=["onboarding"],
        parameters={
            "thermo_input": {
                "temperature": 298.15,
                "pressure": 1.0,
                "simulation_steps": 5000
            }
        }
    )
