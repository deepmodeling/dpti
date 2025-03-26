from prefect import flow

# @flow
# def my_flow(a: str) -> str:
#     return f"Hello, {a=}!"


class GreetingWorkflow:
    def __init__(self):
        pass

    # @flow(log_prints=True, persist_result=True)
    def execute_greeting(self, message: str) -> str:
        """执行问候流程."""
        result = self._process_greeting(message)
        return result

    def _process_greeting(self, message: str) -> str:
        """实际处理问候消息的内部方法."""
        result = f"Hello, {message=}!"
        print(result)
        return result


def create_workflow_flow(workflow_class, method_name, **flow_kwargs):
    """将工作流类方法包装为顶层flow函数.

    Args:
        workflow_class: 工作流类
        method_name: 要包装的方法名
        flow_kwargs: 传递给flow装饰器的参数

    Returns
    -------
    装饰后的flow函数
    """

    @flow(**flow_kwargs)
    def workflow_flow(*args, **kwargs):
        """创建工作流实例并执行指定方法."""
        instance = workflow_class()
        method = getattr(instance, method_name)
        return method(*args, **kwargs)

    return workflow_flow


# 创建顶层flow函数
greeting_flow = create_workflow_flow(
    GreetingWorkflow, "execute_greeting", log_prints=True, name="greeting-workflow"
)


@flow
def my_flow(a: str) -> str:
    workflow_1 = GreetingWorkflow()
    workflow_1.execute_greeting(a)


if __name__ == "__main__":
    # 方法1: 使用包装后的顶层函数
    # 本地运行
    # print(greeting_flow(message="world"))

    # 部署到服务器
    # greeting_flow.serve(
    #     name="greeting-workflow",
    #     parameters={"message": "world"}
    # )

    # 方法2: 直接使用实例的flow方法
    workflow = GreetingWorkflow()
    workflow.execute_greeting.serve(
        name="greeting-workflow", parameters={"message": "world"}
    )
