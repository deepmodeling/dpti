from typing import Callable, Any, Dict, List
from functools import wraps
from abc import ABC, abstractmethod

# abstra
class PrepareOperation(ABC):
    @abstractmethod
    def execute(self, instance: Any, result: Dict[str, Any]) -> None:
        pass

# 具体操作类：上传文件
class UploadFilesOperation(PrepareOperation):
    def execute(self, instance: Any, result: Dict[str, Any]) -> None:
        produced_upload_files = instance.upload_files(io_handler=instance.io_handler)
        result["produced_upload_files"] = produced_upload_files

# 具体操作类：导出设置
class ExportSettingsOperation(PrepareOperation):
    def __init__(self, filename: str = 'settings.json'):
        self.filename = filename

    def execute(self, instance: Any, result: Dict[str, Any]) -> None:
        produced_settings_json = instance.io_handler.write_pure_file(
            file_path=self.filename,
            file_content=instance.runtime_entity.model_dump_json(indent=4)
        )
        result["produced_settings_json"] = produced_settings_json

# 装饰器类
# class AOPPrepare:
#     def __init__(self, operations: List[PrepareOperation]):
#         self.operations = operations

#     def __call__(self, func: Callable) -> Callable:
#         @wraps(func)
#         def wrapper(instance, *args, **kwargs):
#             result = func(instance, *args, **kwargs)
            
#             if not isinstance(result, dict):
#                 result = {"original_result": result}

#             for operation in self.operations:
#                 operation.execute(instance, result)

#             return result

#         return wrapper
    

class AOPDecorator(ABC):
    def __init__(self, operations: List[Operation]):
        self.operations = operations

    @abstractmethod
    def __call__(self, func: Callable) -> Callable:
        pass


# @beforePrepare({
#     upload_files = False,
#     settings = "equi_settings.json"
# })




class BeforeDecorator(AOPDecorator):
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(instance, *args, **kwargs):
            context = {}
            for op in self.operations:
                op.execute(instance, context)
            result = func(instance, *args, **kwargs)
            if isinstance(result, dict):
                result.update(context)
            else:
                result = {"original_result": result, **context}
            return result
        return wrapper




# 使用示例
from prefect import task

class YourClass:
    @task
    @AOPPrepare([
        UploadFilesOperation(),
        ExportSettingsOperation('equi_settings.json')
    ])
    def prepare(self) -> Dict[str, Any]:
        runtime_lammps_entity = EquiLammpsInput.model_construct(**self.runtime_entity.model_dump())
        lmp_str = equi.gen_equi_lammps_input(**runtime_lammps_entity.model_dump())

        produced_file = self.io_handler.write_pure_file(
            file_path='in.lammps',
            file_content=lmp_str)

        all_produced_paths = self.io_handler.all_produced_paths

        return {"all_produced_paths": all_produced_paths}
    
#%%

for field_name,field_info in to_fields.items():
    # 检查目标对象的字段名是否在源对象中存在
    if field_name in from_data:
        if isinstance(to_obj.model_fields_set, set):
            print(f"1:{field_name=} {from_data[field_name]=}")
            setattr(to_obj, field_name, from_data[field_name])
        elif isinstance(to_obj.model_fields_set, dict):
            print(f"2:{field_name=} {from_data[field_name]=}")
            to_obj.model_fields_set[field_name] = from_data[field_name]
        else:
            raise RuntimeError
    # 检查目标对象的字段别名是否在源对象中存在
    elif field_info.validation_alias and field_info.validation_alias in from_data:
        if isinstance(to_obj.model_fields_set, set):
            print(f"3:{field_name=} {from_data[field_info.validation_alias]=}")
            setattr(to_obj, field_name, from_data[field_info.validation_alias])
        elif isinstance(to_obj.model_fields_set, dict):
            print(f"4:{field_info.validation_alias=} {from_data[field_info.validation_alias]=}")
            to_obj.model_fields_set[field_info.validation_alias] = from_data[field_info.validation_alias]
        else:
            raise RuntimeError
    else:
        pass