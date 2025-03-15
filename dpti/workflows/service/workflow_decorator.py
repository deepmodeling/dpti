# %%
import functools
import hashlib
from functools import partial
from typing import Any, Dict, Optional, Union

import simplejson
from prefect import task

# %%
REFRESH_CACHE = False  # Can be set as needed


# from prefect.context import TaskRunContext
# note: this file is COPYED from `prefect/tasks.py` (in offical prefect framework githubsource code)
# and modify method `hash_objects` and `task_input_hash` to `task_input_json_hash`

# hope that task_input_json_hash compatible origin task_input_hash provided by prefect..
_md5 = partial(hashlib.md5, usedforsecurity=False)


def stable_hash(*args: Union[str, bytes], hash_algo=_md5) -> str:
    """Given some arguments, produces a stable 64-bit hash of their contents.

    Supports bytes and strings. Strings will be UTF-8 encoded.

    Args:
        *args: Items to include in the hash.
        hash_algo: Hash algorithm from hashlib to use.

    Returns
    -------
    A hex hash.
    """
    h = hash_algo()
    for a in args:
        if isinstance(a, str):
            a = a.encode()
        h.update(a)
    hash = h.hexdigest()
    print(f"stable_hash: end: {hash=} {args=}")
    return hash


def hash_objects(*args, hash_algo=_md5, **kwargs) -> Optional[str]:
    """Attempt to hash objects by dumping to simplejson JSON."""
    # serializer = JSONSerializer(dumps_kwargs={"for_json":True})
    # print(f"{serializer.dumps_kwargs=}")
    # print(f"hash_objects: {args=} {kwargs=}")

    # we use simplejson only for its `for_json` function: we implement for_json method for the Class SimulationBase (amd maybe others).
    # when simplejson dumps json, it just called the instance's `for_json` method.
    print(f"hash_objects begin: {args=} {kwargs=} {hash_algo=}")
    origin_hash = simplejson.dumps((args, kwargs), for_json=True, sort_keys=True)
    print(f"hash_objects:origin_hash: {origin_hash=}")
    hash = stable_hash(origin_hash, hash_algo=hash_algo)

    # caller_frame = inspect.currentframe().f_back
    # caller_info = inspect.getframeinfo(caller_frame)
    # print(f"hash_objects: caller: {caller_info.function} at {caller_info.filename}:{caller_info.lineno}")

    print(f"hash_objects: end: {hash=} {args=} {kwargs=}")
    return hash


def task_input_json_hash(context: object, arguments: Dict[str, Any]) -> Optional[str]:
    """
    A task cache key implementation which hashes all inputs to the task using a JSON serializer.

    Arguments:
        context: the active `TaskRunContext`
        arguments: a dictionary of arguments to be passed to the underlying task

    Returns
    -------
    a string hash if hashing succeeded, else `None`
    """
    hash = hash_objects(
        # We use the task key to get the qualified name for the task and include the
        # task functions `co_code` bytes to avoid caching when the underlying function
        # changes
        task_key=context.task.task_key,  # pyright: ignore[reportAttributeAccessIssue]
        task_name=context.task.name,  # pyright: ignore[reportAttributeAccessIssue]
        hex_co_code=context.task.fn.__code__.co_code.hex(),  # pyright: ignore[reportAttributeAccessIssue]
        arguments=arguments,
    )
    return hash


class PrefectTaskDecorator:
    """Generic Prefect Task decorator that can decorate classes or functions as tasks.

    Supports multi-level configuration:
    1. Global configuration: Specified when creating the decorator, applies to all decorated objects
    2. Decorator configuration: Specified when applying the decorator, applies to a single decorated object
    3. Method configuration: Specified in the class configuration for specific methods, highest priority
    """

    def __init__(self, **global_task_settings):
        """Initialize decorator instance, set global task configuration.

        Args:
            **global_task_settings: Global settings applicable to all tasks
        """
        self.global_task_settings = global_task_settings

    def __call__(self, obj=None, **decorator_task_settings):
        """Apply decorator to an object, supports direct call and parameterized call.

        Args:
            obj: Object to decorate (function or class)
            **decorator_task_settings: Configuration specific to this decorator call
        """
        if obj is None:
            # Parameterized call (e.g. @decorator(retries=3))
            @functools.wraps(self.__call__)
            def configured_decorator(inner_obj):
                combined_task_settings = {
                    **self.global_task_settings,
                    **decorator_task_settings,
                }
                return self._decorate_object(inner_obj, combined_task_settings)

            return configured_decorator

        # Direct call (e.g. @decorator)
        return self._decorate_object(obj, self.global_task_settings)

    def _decorate_object(self, obj, combined_task_settings):
        """Choose appropriate decoration method based on object type.

        Args:
            obj: Object to decorate
            combined_task_settings: Merged global and decorator-level configuration
        """
        if isinstance(obj, type):
            return self._decorate_class(obj, combined_task_settings)
        else:
            return self._decorate_function(obj, combined_task_settings)

    def _decorate_class(self, cls, combined_task_settings):
        """Decorate specified class methods as Prefect tasks.

        Args:
            cls: Class to decorate
            combined_task_settings: Merged configuration
        """
        # Extract method-specific configuration dictionary
        methods_config_dict = combined_task_settings.get("method_settings", {})
        methods_to_decorate = ["prepare", "run", "extract"]

        # Store original method references
        original_methods = {}
        for method_name in methods_to_decorate:
            if hasattr(cls, method_name):
                original_methods[method_name] = getattr(cls, method_name)

        # Decorate each method
        for method_name, original_method in original_methods.items():
            # Create base configuration for this method
            base_method_settings = {
                "name": f"{cls.__name__}().{method_name}",
                "retries": combined_task_settings.get("retries", 0),
                "log_prints": combined_task_settings.get("log_prints", True),
                "cache_key_fn": task_input_json_hash,
            }

            # Merge method-specific configuration (if exists)
            method_specific_settings = methods_config_dict.get(method_name, {})
            final_method_settings = {**base_method_settings, **method_specific_settings}

            # Create closure to avoid loop variable issues
            def create_wrapped_method(
                method_name, original_method, final_method_settings
            ):
                @task(**final_method_settings)
                @functools.wraps(original_method)
                def wrapped_method(self, *args, **kwargs):
                    print(
                        f"Running {method_name} with settings: {final_method_settings}"
                    )
                    return original_method(self, *args, **kwargs)

                return wrapped_method

            # Replace original method
            setattr(
                cls,
                method_name,
                create_wrapped_method(
                    method_name, original_method, final_method_settings
                ),
            )

        return cls

    def _decorate_function(self, func, combined_task_settings):
        """Decorate function as Prefect task.

        Args:
            func: Function to decorate
            combined_task_settings: Merged configuration
        """
        # Create function task configuration
        function_task_settings = {
            "name": combined_task_settings.get("name", "func_" + func.__name__),
            "retries": combined_task_settings.get("retries", 0),
            "log_prints": combined_task_settings.get("log_prints", True),
            "cache_key_fn": task_input_json_hash,
        }

        # Merge any additional task configuration
        extra_task_settings = combined_task_settings.get("task_settings", {})
        final_function_settings = {**function_task_settings, **extra_task_settings}

        # Apply task decorator
        task_decorator = task(**final_function_settings)

        @task_decorator
        @functools.wraps(func)
        def wrapped_function(*args, **kwargs):
            print(f"Running task: {final_function_settings['name']}")
            return func(*args, **kwargs)

        return wrapped_function


# %%


workflow_task_decorator = PrefectTaskDecorator(
    log_prints=True, retries=0, method_settings={}
)

# %%


@workflow_task_decorator(
    task_configs={
        "prepare": {"name": "process_setup"},
        "run": {"retries": 1},
        "extract": {"name": "process_results"},
    }
)
class DataProcessor:
    def __init__(self, data):
        self.data = data

    def prepare(self):
        print("Setup processing environment")
        return self.data

    def run(self):
        print("Execute data processing")
        self.data["processed"] = True
        return self.data

    def extract(self):
        print("Extract processing results")
        self.data["completed"] = True
        return self.data
