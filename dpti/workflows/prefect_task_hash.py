import hashlib
import simplejson
from typing import Dict, Any, Optional, Union, TypeVar, Protocol
from functools import partial
# from prefect.context import TaskRunContext

# note: this file is COPYED from `prefect/tasks.py` and modify method `hash_objects` and `task_input_hash` to `task_input_json_hash`
# hope that task_input_json_hash compatible origin task_input_hash provided by prefect..
_md5 = partial(hashlib.md5, usedforsecurity=False)


def stable_hash(*args: Union[str, bytes], hash_algo=_md5) -> str:
    """Given some arguments, produces a stable 64-bit hash of their contents.

    Supports bytes and strings. Strings will be UTF-8 encoded.

    Args:
        *args: Items to include in the hash.
        hash_algo: Hash algorithm from hashlib to use.

    Returns:
        A hex hash.
    """
    h = hash_algo()
    for a in args:
        if isinstance(a, str):
            a = a.encode()
        h.update(a)
    return h.hexdigest()

def hash_objects(*args, hash_algo=_md5, **kwargs) -> Optional[str]:
    """
    Attempt to hash objects by dumping to simplejson JSON.
    """
    # serializer = JSONSerializer(dumps_kwargs={"for_json":True})
    # print(f"{serializer.dumps_kwargs=}")

    # print(f"hash_objects: {args=} {kwargs=}")
    origin_hash = simplejson.dumps((args, kwargs), for_json=True, sort_keys=True)
    hash = stable_hash(origin_hash, hash_algo=hash_algo)
    print(f"task_input_json_hash: {hash=} {args=} {kwargs=}")
    return hash

def task_input_json_hash(
    context: object, arguments: Dict[str, Any]
) -> Optional[str]:
    """
    A task cache key implementation which hashes all inputs to the task using a JSON serializer. 

    Arguments:
        context: the active `TaskRunContext`
        arguments: a dictionary of arguments to be passed to the underlying task

    Returns:
        a string hash if hashing succeeded, else `None`
    """
    hash =  hash_objects(
        # We use the task key to get the qualified name for the task and include the
        # task functions `co_code` bytes to avoid caching when the underlying function
        # changes
        context.task.task_key, # pyright: ignore[reportAttributeAccessIssue]
        context.task.fn.__code__.co_code.hex(), # pyright: ignore[reportAttributeAccessIssue]
        arguments,
    )
    return hash