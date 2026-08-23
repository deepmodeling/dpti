import json
import os

from dpdispatcher import Machine, Resources, Submission

_TRANSIENT_DAG_RUN_STATES = {
    "queued",
    "scheduled",
    "running",
    "up_for_retry",
    "up_for_reschedule",
    "deferred",
    "restarting",
}


def is_transient_dag_run_state(state):
    """Return whether an Airflow DAG run should continue being polled."""
    if state is None:
        # A freshly triggered run may not be visible in the metadata DB yet.
        return True
    state_value = getattr(state, "value", state)
    return str(state_value).lower() in _TRANSIENT_DAG_RUN_STATES


def get_empty_submission(job_work_dir, context):
    # context = get_current_context()
    dag_run = context["params"]
    work_base_dir = dag_run["work_base_dir"]
    print("debug781", context)

    with open(os.path.join(work_base_dir, "machine.json")) as f:
        mdata = json.load(f)
    machine = Machine.load_from_dict(mdata["machine"])
    resources = Resources.load_from_dict(mdata["resources"])

    submission = None
    submission = Submission(
        work_base=job_work_dir,
        resources=resources,
        machine=machine,
    )
    return submission
