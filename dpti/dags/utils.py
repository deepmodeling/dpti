from dpdispatcher import Submission, Resources, Machine
import os, json
def get_empty_submission(job_work_dir, context):
    # context = get_current_context()
    dag_run = context['params']
    work_base_dir = dag_run['work_base_dir']
    print('debug781', context)

    with open(os.path.join(work_base_dir, 'machine.json'), 'r') as f:
        mdata = json.load(f)
    machine = Machine.load_from_dict(mdata['machine'])
    resources = Resources.load_from_dict(mdata['resources'])

    submission = None
    submission = Submission(
        work_base=job_work_dir,
        resources=resources,
        machine=machine,
    )
    return submission
