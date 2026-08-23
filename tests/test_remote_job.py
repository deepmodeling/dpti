import unittest
from unittest.mock import MagicMock

from dpti.lib.RemoteJob import SlurmJob, _set_default_resource


class TestRemoteJobResources(unittest.TestCase):
    def test_none_resources_return_defaults(self):
        """Omitted resources become a usable mapping instead of staying None."""
        resources = _set_default_resource(None)

        self.assertEqual(resources["numb_node"], 1)
        self.assertEqual(resources["task_per_node"], 1)
        self.assertFalse(resources["with_mpi"])

    def test_slurm_script_accepts_omitted_resources(self):
        """Scheduler script generation assigns the returned defaults."""
        job = object.__new__(SlurmJob)
        job.remote_root = "/remote/work"
        job.ssh = MagicMock()
        script_file = (
            job.ssh.open_sftp.return_value.open.return_value.__enter__.return_value
        )

        script_name = job._make_script(["task.000000"], "run.sh", res=None)

        self.assertEqual(script_name, "run.sub")
        script = script_file.write.call_args.args[0]
        self.assertIn("#SBATCH -N 1", script)
        self.assertIn("#SBATCH --ntasks-per-node 1", script)


if __name__ == "__main__":
    unittest.main()
