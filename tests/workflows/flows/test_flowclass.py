from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from prefect import flow


class BaseWorkflow(ABC):
    """基础工作流类"""
    
    def __init__(self):
        """初始化工作流，应用装饰器"""
        # 保存原始的execute方法
        
        # 用flow装饰器装饰execute方法
        self.flow_object = flow()(self.execute)
    
    def execute(self, config_yaml: str, flow_trigger_dir: str, flow_run_number: int, **kwargs) -> Dict[str, Any]:
        """执行工作流 - 会被@flow装饰"""
        # 调用子类实现的方法
        return self._execute_impl(config_yaml, flow_trigger_dir, flow_run_number, **kwargs)
    
    @abstractmethod
    def _execute_impl(self, config_yaml: str, flow_trigger_dir: str, flow_run_number: int, **kwargs) -> Dict[str, Any]:
        """子类实现工作流逻辑"""
        pass


class FreeEnergyWorkflow(BaseWorkflow):
    """自由能工作流"""
    
    def _execute_impl(self, config_yaml: str, flow_trigger_dir: str, flow_run_number: int, **kwargs) -> Dict[str, Any]:
        """实现工作流逻辑"""
        # 这里是业务实现
        return {"status": "success"}


flow_object = flow()(FreeEnergyWorkflow().execute)

# @flow
# def test_flowclass():
# 使用示例
if __name__ == "__main__":
    # 创建工作流实例
    workflow = FreeEnergyWorkflow()
    
    # 直接执行工作流
    # result = workflow.execute(
    #     config_yaml="FreeEnergy.yaml",
    #     flow_trigger_dir="./examples",
    #     flow_run_number=0
    # )
    
    # 使用新的API部署工作流
    workflow.flow_object.serve(
        name="dpti-workflow-flowclass-deployment",
        tags=["onboarding"],
        parameters={
            "config_yaml": "FreeEnergy.yaml",
            "flow_trigger_dir": "./examples",
            "flow_run_number": 0
        }
    )