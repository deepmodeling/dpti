"""Tasks for interacting with AWS Batch"""

from typing import Any, Dict, Optional, List

# from dpdispatcher import DPdisTask
from prefect.logging import get_run_logger
from prefect import task
from prefect.utilities.asyncutils import run_sync_in_worker_thread, sync_compatible
# from prefect_aws.credentials import AwsCredentials
from dpdispatcher import Machine as DPispatcherMachine
from dpdispatcher import Task as DPispatcherTask
from dpdispatcher import Resources as DPispatcherResources
from dpdispatcher import Submission as DPispatcherSubmission
# Resources, Submission, Task

default_config = {
    "machine":{
        "batch_type": "Shell",
        "context_type": "LazyLocalContext",
        "local_root": "./",
        # "remote_root": "./tmp_shell_trival_dir"
    },
    "resources":{
        "number_node": 1,
        "cpu_per_node": 4,
        "gpu_per_node": 0,
        "queue_name": "CPU",
        "group_size": 2
    }
}

@task
@sync_compatible
async def dpdispatcher_submit(
    work_base: str,
    # task_list: List[DPispatcherTask],
    # machine: DPispatcherMachine,
    # resources: DPispatcherResources,
    forward_common_files=[],
    backward_common_files=[],
    **kwargs: Optional[Dict[str, Any]],
) -> str:
    """
    Submit a job to the DPDispatcher Batch job service.

    Args:
        job_name: The AWS batch job name.
        job_queue: Name of the AWS batch job queue.
        job_definition: The AWS batch job definition.
        # aws_credentials: Credentials to use for authentication with AWS.
        **batch_kwargs: Additional keyword arguments to pass to the boto3
            `submit_job` function. See the documentation for
            [submit_job](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/batch.html#Batch.Client.submit_job)
            for more details.

    Returns:
        The id corresponding to the job.

    Example:
        Submits a job to batch.

        ```python
        from prefect import flow
        from prefect_aws import AwsCredentials
        from prefect_aws.batch import batch_submit


        @flow
        def example_dpdispatcher_submit_flow():
            # aws_credentials = AwsCredentials(
            #     aws_access_key_id="acccess_key_id",
            #     aws_secret_access_key="secret_access_key"
            # )
            job_id = batch_submit(
                "job_name",
                "job_queue",
                "job_definition",
                aws_credentials
            )
            return job_id

        example_batch_submit_flow()
        ```

    """  # noqa
    logger = get_run_logger()
    logger.info("Preparing to use dpdispatcher to submit to Batch Job system. (Like Slurm, PBS, Bohrium etc)")
    # logger.info("Preparing to submit %s job to %s job queue", job_name, job_queue)

    # batch_client = aws_credentials.get_boto3_session().client("batch")

    # def get_empty_submission(job_work_dir, context):
    # context = get_current_context()
    # dag_run = context["params"]
    # work_base_dir = dag_run["work_base_dir"]
    # print("debug781", context)

    # with open(os.path.join(work_base_dir, "machine.json")) as f:
    #     mdata = json.load(f)

    task1 = DPispatcherTask(
            command="echo 1 >> tmp.txt && sleep 30",
            task_work_path="./",
            forward_files=["example.txt"],
            backward_files=["out.txt"],
            outlog="out.txt",
        )
    task2 = DPispatcherTask(
        command="sleep 60",
        task_work_path="./",
        forward_files=["example.txt"],
        backward_files=["out.txt"],
        outlog="out.txt",
    )

    task_list = [task1, task2]
    machine = DPispatcherMachine(**default_config['machine'])
    resources = DPispatcherResources(**default_config['resources'])
    submission = DPispatcherSubmission(
        work_base=work_base,
        resources=resources,
        machine=machine,
        task_list=task_list
    )

    submission_hash = submission.submission_hash

    print(f"Using dpdispatcher: submission_hash:{submission_hash} generte. Prepare to run")
    # r = sub
    # background_task = asyncio.create_task(submission2.async_run_submission(check_interval=2, clean=False))
    # background_tasks.add(background_task)
    submission_dict = await submission.async_run_submission()
    # result = await asyncio.gather(*background_tasks)

    return submission_hash

    return submission


    response = await run_sync_in_worker_thread(
        batch_client.submit_job,
        jobName=job_name,
        jobQueue=job_queue,
        jobDefinition=job_definition,
        **batch_kwargs,
    )
    return response["jobId"]
