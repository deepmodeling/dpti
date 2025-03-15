#%%
import os
import functools
from collections import defaultdict
from contextlib import contextmanager
from typing import List, Any, Iterator, ClassVar, Optional
from typing import Protocol
from pydantic import BaseModel
from contextvars import ContextVar
from typing import Any
import shutil


#%%

current_io_context: ContextVar[Optional['IOHandler']] = ContextVar('current_io_context', default=None)


def get_current_io() -> Optional['IOHandler']:
    """get current IOHandler, return None if no IOHandler is set"""
    return current_io_context.get()

class FilesToUploadEntity(BaseModel):
    local_files: List[str]
    fields_in_updated_nodedata: List[str]


class IOHandler(Protocol):
    "flow_trigger_dir/flow_running_dir/job_dirname/subjob_dirname/"
    "my_water_example/400K_10000bar_sim/hti_sim/htisubtask_1/"

    flow_trigger_dir: str
    flow_running_dir: str
    job_dirname: str
    job_dir: str
    all_produced_paths: defaultdict
    current_produced_paths: List[str] = []
    def setup_flow_running_dir(self) -> str: ...

    def read_file(self, file_path: str) -> str: ...

    def write_pure_file(self, file_path: str, file_content: str) -> str: ...
    # def upload_files(self, file_paths: List[str], base_dir: str) -> List[str]:
    #     raise NotImplementedError
    def upload_file(self, file_path: str, base_dir: str, new_file_name=None) -> str: # os.path.join(base_dir, file_path)
        raise NotImplementedError

    def jobdir_context(self, job_dirname:str) -> Any:...

    def subjobdir_context(self, subjob_dirname:str='./') -> Any:...

    def isfile(self, file_path: str) -> bool: ...

    def isdir(self, dir_path: str) -> bool: ...

    # @classmethod
    # def get_current_context(cls):
    #     return cls._current_context
    # def __enter__(self) -> Any:
    #     pass
    # def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    #     pass

class LocalFileHandler: # implement IOHandler

    
    def __init__(self, flow_trigger_dir: str = "./",
        flow_running_dirname: str = "./") -> None:
        self.flow_trigger_dir = flow_trigger_dir
        self.flow_running_dirname = flow_running_dirname
        self.flow_running_dir = os.path.join(self.flow_trigger_dir, self.flow_running_dirname)

        self.all_produced_paths = defaultdict(list)
        self.current_produced_paths: List[str] = []
        self.job_dirname = "./"
        self.job_dir = os.path.join(self.flow_running_dir, self.job_dirname)
        self.subjob_dirname = "./"
        # self.setup_flow_running_dir()
        # self.job_dir = self.flow_running_dir
        # self. _current_context = self
        # self.create_job_dir()

    # @staticmethod
    # def ensure_create_job_dir(func):
    #     @functools.wraps(func)
    #     def wrapper(self, *args, **kwargs):
    #         # if not getattr(self, '_job_dir_created', False):q
    #         # if self.job_dir not in self.all_produced_paths:
    #         if not os.path.isdir(self.job_dir):
    #             # print(f"creating job_dir:{self.job_dir=} {self.all_produced_paths=}")
    #             print(f"^^^^^^^^^^^^^^^^^^^^^^^^^creating job_dir:{self.job_dir=}^^^^^^^^^^^^^^^^^^^^^^^^^")
    #             self.create_job_dir()
    #         else:
    #             # print(f"skip create job_dir {self.job_dir=} {self.all_produced_paths=}")
    #             pass
    #         return func(self, *args, **kwargs)
    #     return wrapper
     
    # @classmethod
    # def use_job_info(self, job_dirname: str) -> None:
    #     # self.flow_running_dir = flow_running_dir
    #     self.job_dirname = job_dirname
    #     self.job_dir = os.path.join(self.flow_running_dir,
    #         self.job_dirname)
    def setup_flow_running_dir(self) -> str:
        if os.path.isdir(self.flow_running_dir):
            produced_flow_running_dir = self.flow_running_dir
        else:
            produced_flow_running_dir = self.create_path_with_backup(self.flow_running_dir)
        self.current_produced_paths.append(produced_flow_running_dir)
        return produced_flow_running_dir


        

    def read_file(self, file_path: str) -> str:
        abs_file_path = os.path.join(self.job_dir, file_path)
        with open(abs_file_path, 'r') as f:
            content = f.read()
        return content
    
    # @ensure_create_job_dir
    def write_pure_file(self, file_path: str, file_content: str) -> str:
        abs_file_path = os.path.join(self.job_dir, file_path)
        rel_file_path = os.path.relpath(abs_file_path, start=self.flow_running_dir)
        with open(abs_file_path, 'w') as f:
            f.write(file_content)
        self.current_produced_paths.append(rel_file_path)
        return rel_file_path
    
    # @ensure_create_job_dir
    def upload_file(self, file_path: str, base_dir: str, new_file_name=None) -> str:
        produced_symlink = self.create_relative_symlink_file(
            source_base_dir=base_dir,
            source_file_path=file_path,
            target_dir=self.job_dir,
            new_linkfile_name=new_file_name
        )
        return produced_symlink
    
    # @ensure_create_job_dir
    def upload_files(self, file_paths: List[str], base_dir: str) -> List[str]:
        produced_symlinks = []
        for file_path in file_paths:
            return_link = self.create_relative_symlink_file(
                source_base_dir=base_dir,
                source_file_path=file_path,
                target_dir=self.job_dir,
                new_linkfile_name=None
            )
            produced_symlinks.append(return_link)
        return produced_symlinks

    @contextmanager
    def jobdir_context(self, job_dirname: str) -> Iterator[Any]:
        ori_job_dirname = self.job_dirname
        ori_job_dir = self.job_dir
        # Set context token for possible nested contexts
        token = current_io_context.set(self)
        try:
            # Update the job_dirname and job_dir
            self.job_dirname = job_dirname
            self.job_dir = os.path.join(self.flow_running_dir, job_dirname)
            print(f"LocalFileHandler: Entering job context {self.flow_running_dir=} {job_dirname=} {self.job_dir=}")
            
            # Ensure the directory exists
            if not os.path.isdir(self.job_dir):
                self.job_dir = self.create_path_with_backup(self.job_dir)
                # self.create_job_dir()
                
            yield self
        except Exception as e:
            print(f"LocalFileHandler: An exception occurred in job_dir_context: {e}")
            raise e
        finally:
            # Restore original state
            self.job_dirname = ori_job_dirname
            self.job_dir = ori_job_dir
            current_io_context.reset(token)

    @contextmanager
    def subjobdir_context(self, subjob_dirname: str = "./") -> Iterator[Any]:
        ori_job_dir = self.job_dir
        token = current_io_context.set(self)
        try:
            # Update only the job_dir
            subjob_dir = os.path.join(ori_job_dir, subjob_dirname)
            self.job_dir = subjob_dir
            print(f"LocalFileHandler: Entering subjob_dir context {ori_job_dir=} {subjob_dirname=} {self.job_dir=}")
            # Ensure the directory exists
            if not os.path.isdir(self.job_dir):
                self.job_dir = self.create_path_with_backup(self.job_dir)
            yield self
        except Exception as e:
            print(f"LocalFileHandler: An exception occurred in subdir_context: {e}")
            raise e
        finally:
            # Restore original state
            self.job_dir = ori_job_dir
            current_io_context.reset(token)
    
    def isfile(self, file_path: str) -> bool:
        is_file = os.path.isfile(os.path.join(self.job_dir, file_path))
        return is_file

    def isdir(self, dir_path: str) -> bool:
        is_dir = os.path.isdir(os.path.join(self.job_dir, dir_path))
        return is_dir

    def create_relative_symlink_file(self, source_base_dir, source_file_path,
            target_dir, new_linkfile_name=None):
        abs_file_path = os.path.join(source_base_dir, source_file_path)
        if not os.path.isfile(abs_file_path):
            raise RuntimeError(f"{os.getcwd()=}, {abs_file_path=} must be a file. "
                               f"{source_file_path=}. {target_dir=} {source_base_dir=} {new_linkfile_name=}")
        # file_abs_path = os.path.abspath(file_path)
        file_basename = os.path.basename(source_file_path)

        abs_target_dir = os.path.join(source_base_dir, target_dir)
        relative_path = os.path.relpath(abs_file_path, start=abs_target_dir)
        if new_linkfile_name is None:
            target_linkfile_path = os.path.join(abs_target_dir, file_basename)
        else:
            target_linkfile_path = os.path.join(abs_target_dir, new_linkfile_name)
        os.symlink(src=relative_path, dst=target_linkfile_path)

        rel_target_linkfile_path = os.path.relpath(target_linkfile_path, start=self.flow_running_dir)
        return rel_target_linkfile_path
    
    @staticmethod
    def create_path_with_backup(path: str) -> str:
        path += "/"
        if os.path.isdir(path):
            dirname = os.path.dirname(path)
            counter = 0
            while True:
                bk_dirname = dirname + ".bk%03d" % counter
                if not os.path.isdir(bk_dirname):
                    shutil.move(dirname, bk_dirname)
                    break
                counter += 1
        os.makedirs(path)
        abs_path = os.path.abspath(path)
        return abs_path

#%%

