# import injector
import typing
from contextlib import contextmanager
from typing import Callable, TypeVar, Union

from injector import Injector


class InjectionContext:
    _contexts = []

    def __init__(self, di_container: Injector):
        self.di_container = di_container

    @classmethod
    def get_current(cls):
        return cls._contexts[-1] if cls._contexts else None

    def __enter__(self):
        self.__class__._contexts.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__class__._contexts.pop()


@contextmanager
def injection_context(di_container: Injector):
    with InjectionContext(di_container):
        yield


T = TypeVar("T")


def context_inject(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator function to inject dependencies based on function parameter type annotations.
    This decorator function is used to inject dependencies based on function parameter type annotations.
    It will try to get the dependency from the dependency injection container and inject it into the function.
    If the dependency is not found, it will raise an exception.

    Example:
        @context_inject
        def my_function(param1: MyClass, param2: int) -> None:
            pass

    In this example, `param1` will be injected with an instance of `MyClass` from the dependency injection container,
        and `param2` will be injected with an instance of `int` from the dependency injection container.

    Args:
        func (Callable[..., T]): Decorated function.

    Returns
    -------
    Callable[..., T]:  Wrapper function that injects dependencies based on function parameter type annotations.
    """

    def wrapper(*args, **kwargs):
        context = InjectionContext.get_current()
        print(f"context:{context=}")
        if context:
            di_container = context.di_container
            # get function parameters typing annotations
            annotations = func.__annotations__
            print(f"context_inject: {annotations=}")
            for param_name, param_type in annotations.items():
                print(f"context_inject: {param_name=}, {param_type=}")
                if param_name not in kwargs and param_name != "return":
                    if typing.get_origin(param_type) is Union and type(
                        None
                    ) in typing.get_args(param_type):
                        # means Optional[Any]. for example: Optional[List[str]],  Union[float, None]  both will not trigger inject
                        pass
                    else:
                        try:
                            # try to get dependency from injector
                            kwargs[param_name] = di_container.get(param_type)
                        except Exception as e:
                            print(
                                f"{param_type=} {dir(param_type)=} {type(param_type)=} "
                            )
                            print(
                                f"context inject fail!  {param_name=} {param_type=} {context=} {di_container=} {annotations=}"
                            )
                            raise e
                        # pass  # keep original if fail
        return func(*args, **kwargs)

    return wrapper


class InjectableMeta(type):
    pass


# %%
