import functools
from typing import Type, Callable, Dict, Any, Union, Optional
from prefect import task, flow

class PrefectDecorator:
    """通用的Prefect装饰器，可以装饰类或函数为task"""
    
    def __init__(self, **common_configs):
        self.common_configs = common_configs
    
    def __call__(self, obj=None, **specific_configs):
        if obj is None:
            @functools.wraps(self.__call__)
            def configured_decorator(inner_obj):
                merged_configs = {**self.common_configs, **specific_configs}
                return self._decorate_object(inner_obj, merged_configs)
            return configured_decorator
        
        return self._decorate_object(obj, self.common_configs)
    
    def _decorate_object(self, obj, configs):
        if isinstance(obj, type):
            return self._decorate_class(obj, configs)
        else:
            return self._decorate_function(obj, configs)
    
    def _decorate_class(self, cls, configs):
        task_configs = configs.get('task_configs', {})
        methods_to_decorate = ['prepare', 'run', 'extract']
        
        original_methods = {}
        for method_name in methods_to_decorate:
            if hasattr(cls, method_name):
                original_methods[method_name] = getattr(cls, method_name)
        
        for method_name, original_method in original_methods.items():
            method_configs = {
                'name': f"{cls.__name__}.{method_name}",
                'retries': configs.get('retries', 0),
                'log_prints': configs.get('log_prints', True)
            }
            
            if method_name in task_configs:
                method_configs.update(task_configs[method_name])
            
            # 创建闭包避免循环变量问题
            def create_wrapped_method(method_name, original_method, method_configs):
                @task(**method_configs)
                @functools.wraps(original_method)
                def wrapped_method(self, *args, **kwargs):
                    print(f"Running {method_name} with configs {method_configs}")
                    return original_method(self, *args, **kwargs)
                return wrapped_method
            
            setattr(cls, method_name, create_wrapped_method(method_name, original_method, method_configs))
        
        return cls
    
    def _decorate_function(self, func, configs):
        name = configs.get('name', func.__name__)
        retries = configs.get('retries', 0)
        log_prints = configs.get('log_prints', True)
        
        task_configs = {
            'name': name,
            'retries': retries,
            'log_prints': log_prints,
            **configs.get('task_config', {})
        }
        
        decorator = task(**task_configs)
        
        @decorator
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            print(f"Running task: {name}")
            return func(*args, **kwargs)
        
        return wrapped


# 创建装饰器实例
mydec = PrefectDecorator(log_prints=True)

# 装饰测试函数和类
@mydec(name="data_prep")
def prepare_data(input_data):
    print("准备数据")
    return {"prepared": True, "data": input_data}

@mydec(task_configs={
    "prepare": {"name": "process_setup"},
    "run": {"retries": 2},
    "extract": {"name": "process_results"}
})
class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    def prepare(self):
        print("设置处理环境")
        return self.data
    
    def run(self):
        print("执行数据处理")
        self.data["processed"] = True
        return self.data
    
    def extract(self):
        print("提取处理结果")
        self.data["completed"] = True
        return self.data

# 定义工作流
@flow(name="data_processing_flow")
def process_workflow(input_data):
    prepared_data = prepare_data(input_data)
    processor = DataProcessor(prepared_data)
    processor.prepare()
    processor.run()
    result = processor.extract()
    return result

# 简单测试
def test_basics():
    print("\n=== 基本功能测试 ===")
    # 测试装饰函数
    result = prepare_data({"test": True})
    print(f"函数结果: {result}")
    
    # 测试装饰类
    processor = DataProcessor({"input": "value"})
    processor.prepare()
    processor.run()
    class_result = processor.extract()
    print(f"类方法结果: {class_result}")
    
    # 测试完整工作流
    flow_result = process_workflow({"new": "data"})
    print(f"工作流结果: {flow_result}")
    print("=== 测试完成 ===\n")

# 主函数
if __name__ == "__main__":
    import sys
    
    test_basics()
    process_workflow.serve(
        name="data-processor",
        parameters={"input_data": {"default": "data"}}
    )
    # 否则直接运行工作流
