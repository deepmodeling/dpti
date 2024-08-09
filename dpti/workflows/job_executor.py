from typing import Protocol, Any, Optional
import os
import glob

from dpdispatcher import Job, Machine as DPDispatcherMachine
from dpdispatcher import Task as DPDispatcherTask
from dpdispatcher import Resources as DPDispatcherResources
from dpdispatcher import Submission as DPDispatcherSubmission

class JobExecutor(Protocol):
    def submit(self, job_dir: str) -> Any:
        raise NotImplementedError
    
    def group_submit(self, job_dir: str, subtasks_template:str = "", command:str = "") -> Any:
        raise NotImplementedError

class DpdispatcherExecutor: # implements JobExecutor
    job_dir: str
    submission: DPDispatcherSubmission

    default_config = {
    "machine":{
        "batch_type": "DpCloudServer",
        "context_type": "DpCloudServerContext",
        "local_root" : "./",
        "remote_profile":{
            "email": "1109111326@qq.com",
            "password": "pengqiong@123",
            "program_id": 12816,
            "keep_backup": False,
              "input_data":{
                  "api_version":2,
                  "job_type": "container",
                  "log_file": "*/log.lammps",
                  "grouped": True,
                #   "checkpoint_time":5,
                  "job_name": "dpti_test_V100",
                #   "disk_size": 100,
                  "scass_type":"1 * NVIDIA GPU_16g",
                  "platform": "ali",
                  "image_name":"registry.dp.tech/dptech/deepmd-kit:2.1.5-cuda11.6",
                  "on_demand":0
              }
          }
    },
    "resources": {
        "number_node": 1,
        "cpu_per_node": 4,
        "gpu_per_node": 1,
        "queue_name": "GPU V100",
        "group_size": 40
      }
}

    def __init__(self):

        self.machine = DPDispatcherMachine.load_from_dict(self.default_config['machine'])
        self.resources = DPDispatcherResources.load_from_dict(self.default_config['resources'])

        pass

    # def provide_ex

    # def
    def group_submit(self, job_dir:str, subtasks_template:str="./*/task*", command="lmp -i in.lammps") -> str:
        task_abs_dir_list = glob.glob(os.path.join(job_dir, subtasks_template))

        task_dir_list = [
            os.path.relpath(subdir, start=job_dir) for subdir in task_abs_dir_list
        ]

        # due to Python ZipFile not support symlink
        # We may create it manually during task execution for some cases.
        # pre_command_symlink = "test -f graph.pb || ln -s ../../graph.pb ./; "

        task_list = [DPDispatcherTask(
            command=command,
            task_work_path=subdir,
            # forward_files=["in.lammps", "*lmp", "graph.pb"],
            forward_files=["in.lammps", "*lmp"],
            backward_files=["log.lammps"],
        ) for subdir in task_dir_list
        ]

        self.submission = DPDispatcherSubmission(
            work_base=job_dir,
            resources=self.resources,
            machine=self.machine,
            forward_common_files=["graph.pb", "*lmp"],
            task_list=task_list
        )

        self.submission.generate_jobs()
        print(f"note: submission {self.submission.submission_hash} to be submit")
        submission_r = self.submission.run_submission(check_interval=15)
        submission_hash = str(self.submission.submission_hash)
        return submission_hash

    def submit(self, job_dir) -> str:

        dpdispatcher_task = DPDispatcherTask(
            # command="lmp -pk gpu 1 -sf gpu -i in.lammps",
            command="lmp -i in.lammps",
            task_work_path="./",
            forward_files=["in.lammps", "*lmp", "graph.pb"],
            backward_files=["log.lammps", "dump.equi", "out.lmp"],
        )

        # dpdispatcher_task_list = 

        self.submission = DPDispatcherSubmission(
            work_base=job_dir,
            resources=self.resources,
            machine=self.machine,
            task_list=[dpdispatcher_task]
        )
        # self.submission = submission
        self.submission.generate_jobs()
        print(f"note: submission {self.submission.submission_hash} to be submit")
        submission_r = self.submission.run_submission(check_interval=15)
        submission_hash = str(self.submission.submission_hash)
        return submission_hash
        # print(f"Submitting job: {job_dir}")