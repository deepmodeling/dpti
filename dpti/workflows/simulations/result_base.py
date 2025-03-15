from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, root_validator

class ResultCompleteness(str, Enum):
    """Result completeness status"""
    COMPLETE = "complete"          # All expected results available
    PARTIAL = "partial"            # Some results missing
    PRELIMINARY = "preliminary"    # Initial results, may be updated
    FAILED = "failed"             # Calculation failed

# class ExperimentType(str, Enum):
#     """Type of experiment or analysis"""
#     SIMULATION = "simulation"      # MD simulation
#     ANALYSIS = "analysis"         # Data analysis
#     POST_PROCESS = "post_process" # Post-processing
#     VISUALIZATION = "visualization" # Visualization task

class ResultMetaBase(BaseModel):
    """
    Essential metadata for any result
    
    This is the minimal required metadata that any result should have,
    regardless of its source or type
    """
    # Core identification
    uuid: UUID = Field(default_factory=uuid4)
    result_name: str = Field(..., description="Human readable identifier Usually as variable name in python code")
    result_classname: str = Field(..., description="Classname of the result")
    type_version: str = Field(default="0.1.0", description="Version of the result type")

    # Time tracking
    created_time: datetime = Field(default_factory=datetime.now)
    modified_time: datetime = Field(default_factory=datetime.now)  
    # Type information
    # result_type: str = Field(..., description="Type of result (user-defined)")
    
    
    # Status
    completeness: ResultCompleteness = ResultCompleteness.COMPLETE
    # experiment_type: ExperimentType = ExperimentType.SIMULATION
    
    # Basic provenance
    creator: str = Field(default="unknown")
    # parent_uuid: Optional[UUID] = None
    
    # Custom metadata (user-defined)
    custom_meta: Dict[str, Any] = Field(default_factory=dict)

class ArtifactInfo(BaseModel):
    """Information about result artifacts (plots, large data files etc)"""
    path: Optional[Path] = None
    url: Optional[str] = None
    base64_data: Optional[str] = None
    mime_type: str = "text/plain"
    description: str = "No description"
    size: Optional[int] = None
    
    # @root_validator
    # def check_data_source(cls, values):
    #     """Ensure at least one data source is provided"""
    #     if not any([values.get('path'), values.get('url'), values.get('base64_data')]):
    #         raise ValueError("At least one of path, url or base64_data must be provided")
    #     return values

class ResultDataBase(BaseModel):
    """the class for all result data"""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # metatype: ResultMetaBase
    # result_data: Optional[Any] = None

    artifacts: List[ArtifactInfo] = Field(default_factory=list)
    
    # extra_data: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, result_name: str="default_result_name", **kwargs):
        meta = ResultMetaBase(
            result_name=result_name,
            result_classname=self.__class__.__name__
        )
        super().__init__(meta=meta, **kwargs)

    def save_to_json(self, filepath: Optional[Path] = None) -> None:
        """save the result data to a json file"""
        if filepath is None:
            filepath = self.job_running_dir / "result_data.json"
        filepath.write_text(self.model_dump_json(indent=2))




# 使用示例
def example_usage():
    # 创建结果实例
    result = ResultDataBase(
        result_name="my_simulation",
        # result_data={
        #     "temperatures": [300, 400, 500],
        #     "pressures": [1.0, 1.0, 1.0]
        # },
    )
    
    # 添加图片
    result.add_artifact(
        name="temperature_plot",
        description="Temperature variation plot",
        mime_type="image/png",
        path=Path("./plots/temp.png")
    )
    
    # 添加base64编码的图片
    result.add_artifact(
        name="pressure_plot",
        description="Pressure variation plot",
        mime_type="image/png",
        base64_data="base64_encoded_string_here"
    )
    
    return result
