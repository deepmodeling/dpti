# %%

# %%


import os

# %%
import warnings
from pathlib import Path
from typing import Any, Type, TypeVar

import yaml
from pydantic import BaseModel

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

unvalid_BaseModel_T = TypeVar("unvalid_BaseModel_T", bound=BaseModel)  #


def parse_yaml_like_input(
    value: Any,
    model_class: Type[unvalid_BaseModel_T],
) -> unvalid_BaseModel_T:
    """Generic parser that converts YAML-like inputs to a specified model instance.

    Args:
        value: Input value (model instance, dict, YAML string, or file path)
        model_class: Target model class

    Returns
    -------
    An instance of the requested model class with shallow validation

    Raises
    ------
    ValueError: When parsing fails
    """
    # Case 1: Already the target model instance
    if isinstance(value, model_class):
        return value

    # 获取原始字典
    raw_dict = None

    # Case 2: Dictionary input
    if isinstance(value, dict):
        raw_dict = value
    # Case 3: String or Path input - convert to string immediately
    elif isinstance(value, (str, Path)):
        path_str = str(value)

        # Handle file path
        if path_str.endswith((".yaml", ".yml", ".json")):
            if not os.path.exists(path_str):
                raise FileNotFoundError(f"YAML file not found: {path_str=}")

            with open(path_str) as f:
                raw_dict = yaml.safe_load(f)

            if not isinstance(raw_dict, dict):
                raise ValueError(
                    f"YAML file does not contain a dictionary: {path_str=}"
                )
        else:
            raw_dict = yaml.safe_load(path_str)
            if not isinstance(raw_dict, dict):
                raise ValueError("YAML content is not a dictionary")
    else:
        raise ValueError(f"Cannot parse {value=} as {model_class.__name__=}")

    # 处理第一层嵌套字段
    model_fields = model_class.model_fields
    processed_dict = {}

    for field_name, field_value in raw_dict.items():
        if field_name in model_fields:
            field_type = model_fields[field_name].annotation
            # 检查字段类型是否是Pydantic模型且值是字典
            if (
                hasattr(field_type, "model_construct")
                and isinstance(field_value, dict)
                and getattr(field_type, "__origin__", None) != dict
            ):  # 排除Dict类型
                # 对嵌套对象使用model_construct
                processed_dict[field_name] = field_type.model_construct(**field_value)  # type: ignore[reportOptionalMemberAccess]
            else:
                # 保留原始值
                processed_dict[field_name] = field_value
        else:
            # 未知字段，直接保留
            processed_dict[field_name] = field_value

    # 创建未完成验证的模型实例
    model_obj = model_class.model_construct(**processed_dict)
    return model_obj
