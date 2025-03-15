import glob
import os
from typing import Any, Protocol

from dpdispatcher import Machine as DPDispatcherMachine
from dpdispatcher import Resources as DPDispatcherResources
from dpdispatcher import Submission as DPDispatcherSubmission
from dpdispatcher import Task as DPDispatcherTask

from dpti.workflows.service.dpdispatcher_configs import default_config


class JobExecutor(Protocol):
    def submit(self, job_dir: str, command: str = "") -> Any:
        raise NotImplementedError

    async def group_submit(
        self, job_dir: str, subtasks_template: str = "", command: str = ""
    ) -> Any:
        raise NotImplementedError


class DpdispatcherExecutor:  # implements JobExecutor
    job_dir: str
    submission: DPDispatcherSubmission
    default_config = default_config

    def __init__(self):
        self.machine = DPDispatcherMachine.load_from_dict(
            self.default_config["machine"]
        )
        self.resources = DPDispatcherResources.load_from_dict(
            self.default_config["resources"]
        )
        pass

    async def group_submit(
        self,
        job_dir: str,
        subtasks_template: str = "./*/task*",
        command="lmp -i in.lammps",
    ) -> Any:
        if job_dir is None or job_dir == "":
            raise ValueError("job_dir cannot be None or empty")
        if not os.path.exists(job_dir):
            raise FileNotFoundError(f"Directory does not exist: {job_dir=}")

        task_abs_dir_list = glob.glob(os.path.join(job_dir, subtasks_template))

        task_dir_list = [
            os.path.relpath(subdir, start=job_dir) for subdir in task_abs_dir_list
        ]

        # due to Python ZipFile not support symlink
        # We may create it manually during task execution for some cases.
        # pre_command_symlink = "test -f graph.pb || ln -s ../../graph.pb ./; "

        task_list = [
            DPDispatcherTask(
                command=command,
                task_work_path=subdir,
                # forward_files=["in.lammps", "*lmp", "graph.pb"],
                forward_files=["in.lammps", "*lmp"],
                backward_files=["log.lammps", "*lmp"],
            )
            for subdir in task_dir_list
        ]

        self.submission = DPDispatcherSubmission(
            work_base=job_dir,
            resources=self.resources,
            machine=self.machine,
            forward_common_files=["graph.pb", "*lmp"],
            task_list=task_list,
        )

        self.submission.generate_jobs()
        print(f"note: submission {self.submission.submission_hash} to be submit")
        # submission_r = self.submission.run_submission(check_interval=15)
        # submission_hash = str(self.submission.submission_hash)

        # loop = asyncio.get_event_loop()
        submission_hash = await self.submission.async_run_submission(
            check_interval=60, clean=False
        )

        return submission_hash

    def submit(self, job_dir, command="lmp -i in.lammps") -> str:
        if job_dir is None or job_dir == "":
            raise ValueError("job_dir cannot be None or empty")
        if not os.path.exists(job_dir):
            raise FileNotFoundError(f"Directory does not exist: {job_dir=}")

        dpdispatcher_task = DPDispatcherTask(
            # command="lmp -pk gpu 1 -sf gpu -i in.lammps",
            command=command,
            task_work_path="./",
            forward_files=["in.lammps", "*lmp", "graph.pb"],
            backward_files=["log.lammps", "dump.*", "out.lmp"],
        )

        # dpdispatcher_task_list =

        self.submission = DPDispatcherSubmission(
            work_base=job_dir,
            resources=self.resources,
            machine=self.machine,
            task_list=[dpdispatcher_task],
        )
        # self.submission = submission
        self.submission.generate_jobs()
        print(
            f"note: DpdispatcherExecutor.submit: submission {self.submission.submission_hash} to be submit"
        )
        submission_r = self.submission.run_submission(check_interval=15)
        print(f"note: DpdispatcherExecutor.submit: submission {submission_r} finished")
        submission_hash = str(self.submission.submission_hash)
        return submission_hash
        # print(f"Submitting job: {job_dir}")


# %%
