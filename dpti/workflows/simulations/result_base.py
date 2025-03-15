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
    artifacts: List[ArtifactInfo] = Field(default_factory=list)

    def __init__(self, result_name: str="default_result_name", **kwargs):
        meta = ResultMetaBase(
            result_name=result_name,
            result_classname=self.__class__.__name__
        )
        super().__init__(meta=meta, **kwargs)

    def save_to_json(self, filepath: Optional[Path] = None) -> None:
        pass

    def add_artifact(self, artifact: ArtifactInfo) -> None:
        pass




def example_usage():
    result = ResultDataBase(
        result_name="my_simulation",
    )
    
    result.add_artifact(
        ArtifactInfo(
            path=Path("./plots/temp.png"),
            description="Temperature variation plot",
            mime_type="image/png",
        )
    )
    
    result.add_artifact(
        ArtifactInfo(
            path=Path("./plots/press.png"),
            description="Pressure variation plot",
            mime_type="image/png",
        )
    )
    
    return result
