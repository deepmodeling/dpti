import os
import functools
from collections import defaultdict
from contextlib import contextmanager
from typing import List, Any, Iterator
from typing import Protocol
from pydantic import BaseModel

from ...lib.utils import create_path
# from ...lib.utils import create_relative_symlink_file



class FilesToUploadEntity(BaseModel):
    local_files: List[str]
    fields_in_updated_nodedata: List[str]


class IOHandler(Protocol):
    flow_trigger_dir: str
    flow_running_dir: str
    job_dirname: str
    job_dir: str
    all_produced_paths: defaultdict
    current_produced_paths: List[str] = []

    def use_job_info(self, job_dirname:str) -> Any:
        pass
    def write_pure_file(self, file_path: str, file_content: str) -> str:
        raise NotImplementedError
    def upload_files(self, file_paths: List[str], base_dir: str) -> List[str]:
        raise NotImplementedError
    
    def subdir_context(self, subdirname:str='./') -> Any:
        pass
    # def __enter__(self) -> Any:
    #     pass
    # def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    #     pass


class LocalFileHandler: # implement IOHandler
    
    def __init__(self, flow_trigger_dir: str = "./", flow_running_dir: str = "./") -> None:
        self.flow_trigger_dir = flow_trigger_dir
        self.flow_running_dir = flow_running_dir

        # self.related_workflow = realated_workflow
        # self._job_dir_created: bool = False
        self.all_produced_paths = defaultdict(list)
        self.current_produced_paths: List[str] = []
        self.job_dirname = "default_job/"
        self.job_dir = self.flow_running_dir

    @staticmethod
    def ensure_create_job_dir(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # if not getattr(self, '_job_dir_created', False):q
            # if self.job_dir not in self.all_produced_paths:
            if not os.path.isdir(self.job_dir):
                # print(f"creating job_dir:{self.job_dir=} {self.all_produced_paths=}")
                print(f"^^^^^^^^^^^^^^^^^^^^^^^^^creating job_dir:{self.job_dir=}^^^^^^^^^^^^^^^^^^^^^^^^^")
                self.create_job_dir()
            else:
                # print(f"skip create job_dir {self.job_dir=} {self.all_produced_paths=}")
                pass
            return func(self, *args, **kwargs)
        return wrapper
     
    # @classmethod
    def use_job_info(self, job_dirname: str) -> None:
        # self.flow_running_dir = flow_running_dir
        self.job_dirname = job_dirname
        self.job_dir = os.path.join(self.flow_running_dir, self.job_dirname)

    def create_job_dir(self) -> str:
        self.job_dir = create_path(self.job_dir)
        # self.current_produced_paths.append(self.job_dir)
        self.current_produced_paths = self.all_produced_paths[self.job_dir]
        self.current_produced_paths = list()
        # self._job_dir_created = True
        return self.job_dir
    
    @ensure_create_job_dir
    def write_pure_file(self, file_path: str, file_content: str) -> str:
        abs_file_path = os.path.join(self.job_dir, file_path)
        with open(abs_file_path, 'w') as f:
            f.write(file_content)
        self.current_produced_paths.append(abs_file_path)
        return abs_file_path
    
    @ensure_create_job_dir
    def upload_files(self, file_paths: List[str], base_dir: str) -> List[str]:
        produced_symlinks = self.link_files(link_files=file_paths, base_dir=base_dir)
        return produced_symlinks

    @ensure_create_job_dir
    def link_files(self, link_files: List[str], base_dir: str) -> List[str]:
        produced_symlinks = []
        for file_path in link_files:
            # abs_file_path = os.path.join(self.flow_trigger_dir, file_path)
            target_linkfile_path = self.create_relative_symlink_file(
                file_path=file_path,
                target_dir=self.job_dir,
                work_base_dir=base_dir
            )
            produced_symlinks.append(target_linkfile_path)
        self.current_produced_paths.extend(produced_symlinks)
        return produced_symlinks
    
    @contextmanager
    def subdir_context(self, subdirname: str = "./") -> Iterator[Any]:
        ori_job_dir = self.job_dir
        try:
            # self.job_dir = job_dir
            self.job_dir = os.path.join(ori_job_dir, subdirname)
            print(f"Entering context {ori_job_dir=} {subdirname=} {self.job_dir=}")
            yield self
        except Exception as e:
            print(f"An exception occurred: {e}")
            raise e
        finally:
            self.job_dir = ori_job_dir
    
    # def __enter__(self) -> 'LocalFileHandler':
    #     # self.job_dirname = job_dirname
    #     # self.job_dir = os.path.join(self.flow_running_dir, self.job_dirname)

    #     print(f"Entering context for {self.flow_running_dir} {self.job_dir}")
    #     return self

    # def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    #     print(f"Exiting context for {self.flow_running_dir} {self.job_dir}")
    #     # 在这里可以进行清理操作，比如关闭文件、释放资源等
    #     if exc_type is not None:
    #         print(f"An exception occurred: {exc_val}")
#%%
    @staticmethod
    def create_relative_symlink_file(file_path, target_dir, work_base_dir):
        abs_file_path = os.path.join(work_base_dir, file_path)
        if not os.path.isfile(abs_file_path):
            raise RuntimeError(f"{os.getcwd()=} {abs_file_path=} must be a file.{file_path=}. {target_dir=} {work_base_dir=}")
        # file_abs_path = os.path.abspath(file_path)
        file_basename = os.path.basename(file_path)

        abs_target_dir = os.path.join(work_base_dir, target_dir)
        relative_path = os.path.relpath(abs_file_path, start=abs_target_dir)
        target_linkfile_path = os.path.join(abs_target_dir, file_basename)
        os.symlink(src=relative_path, dst=target_linkfile_path)
        return target_linkfile_path
