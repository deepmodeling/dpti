import enum
import os
import re
import sys
import urllib.parse
import warnings
from typing import Any, Dict, Generator, List, Optional, Tuple

import anyio.abc
import docker
import docker.errors
import packaging.version
from docker import DockerClient
from docker.models.containers import Container
from pydantic import AfterValidator, Field
from slugify import slugify
from typing_extensions import Annotated, Literal

import prefect
from prefect.client.orchestration import ServerType, get_client
from prefect.client.schemas import FlowRun
from prefect.events import Event, RelatedResource, emit_event
from prefect.exceptions import InfrastructureNotAvailable, InfrastructureNotFound
from prefect.server.schemas.core import Flow
from prefect.server.schemas.responses import DeploymentResponse
from prefect.settings import PREFECT_API_URL
from prefect.utilities.asyncutils import run_sync_in_worker_thread
from prefect.utilities.dockerutils import (
    format_outlier_version_name,
    get_prefect_image_name,
    parse_image_tag,
)
from prefect.workers.base import BaseJobConfiguration, BaseWorker, BaseWorkerResult
# from prefect_docker.credentials import DockerRegistryCredentials

from dpdispatcher import Machine as DPispatcherMachine
from dpdispatcher import Task as DPispatcherTask
from dpdispatcher import Resources as DPispatcherResources
from dpdispatcher import Submission as DPispatcherSubmission


class DockerWorkerJobConfiguration(BaseJobConfiguration):
    pass

class DockerWorker(BaseWorker):
    """Prefect worker that executes flow runs within Docker containers."""

    type = "docker"
    job_configuration = DockerWorkerJobConfiguration
    _description = (
        "Execute flow runs within Docker containers. Works well for managing flow "
        "execution environments via Docker images. Requires access to a running "
        "Docker daemon."
    )
    _display_name = "Docker"
    _documentation_url = "https://prefecthq.github.io/prefect-docker/worker/"
    _logo_url = "https://images.ctfassets.net/gm98wzqotmnx/2IfXXfMq66mrzJBDFFCHTp/6d8f320d9e4fc4393f045673d61ab612/Moby-logo.png?h=250"  # noqa

    def __init__(self, *args: Any, test_mode: bool = None, **kwargs: Any) -> None:
        # if test_mode is None:
        #     self.test_mode = bool(os.getenv("PREFECT_DOCKER_TEST_MODE", False))
        # else:
        #     self.test_mode = test_mode
        super().__init__(*args, **kwargs)

    async def setup(self):
        # if not self.test_mode:
        #     self._client = get_client()
        #     if self._client.server_type == ServerType.EPHEMERAL:
        #         raise RuntimeError(
        #             "Docker worker cannot be used with an ephemeral server. Please set"
        #             " PREFECT_API_URL to the URL for your Prefect API instance. You"
        #             " can use a local Prefect API instance by running `prefect server"
        #             " start`."
        #         )
        return await super().setup()

    async def run(
        self,
        flow_run: "FlowRun",
        configuration: BaseJobConfiguration,
        task_status: Optional[anyio.abc.TaskStatus] = None,
    ) -> BaseWorkerResult:
        """
        Executes a flow run within a Docker container and waits for the flow run
        to complete.
        """
        # The `docker` library uses requests instead of an async http library so it must
        # be run in a thread to avoid blocking the event loop.
        # container, created_event = await run_sync_in_worker_thread(
        #     self._create_and_start_container, configuration
        # )
        # container_pid = self._get_infrastructure_pid(container_id=container.id)

        # # Mark as started and return the infrastructure id
        # if task_status:
        #     task_status.started(container_pid)

        # # Monitor the container
        # container = await run_sync_in_worker_thread(
        #     self._watch_container_safe, container, configuration, created_event
        # )

        # exit_code = container.attrs["State"].get("ExitCode")

        machine = DPispatcherMachine.load_from_dict(machine)
        resources = DPispatcherResources.load_from_dict(resources)
        submission = DPispatcherSubmission(
            work_base='./',
            resources=resources,
            machine=machine,
            task_list=task_list
        )

        # return DockerWorkerResult(
        #     status_code=exit_code if exit_code is not None else -1,
        #     identifier=container_pid,
        # )

    async def kill_infrastructure(
        self,
        infrastructure_pid: str,
        configuration: DockerWorkerJobConfiguration,
        grace_seconds: int = 30,
    ):
        """
        Stops a container for a cancelled flow run based on the provided infrastructure
        PID.
        """
        docker_client = self._get_client()

        base_url, container_id = self._parse_infrastructure_pid(infrastructure_pid)
        if docker_client.api.base_url != base_url:
            raise InfrastructureNotAvailable(
                "".join(
                    [
                        (
                            f"Unable to stop container {container_id!r}: the current"
                            " Docker API "
                        ),
                        (
                            f"URL {docker_client.api.base_url!r} does not match the"
                            " expected "
                        ),
                        f"API base URL {base_url}.",
                    ]
                )
            )
        await run_sync_in_worker_thread(
            self._stop_container, container_id, docker_client, grace_seconds
        )