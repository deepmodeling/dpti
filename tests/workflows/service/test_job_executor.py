import unittest
from unittest.mock import Mock, patch
import os
from pathlib import Path

from dpti.workflows.service.job_executor import DpdispatcherExecutor

class TestDpdispatcherExecutor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.patcher_machine = patch('dpti.workflows.service.job_executor.DPDispatcherMachine')
        self.patcher_resources = patch('dpti.workflows.service.job_executor.DPDispatcherResources')
        self.patcher_task = patch('dpti.workflows.service.job_executor.DPDispatcherTask')
        self.patcher_submission = patch('dpti.workflows.service.job_executor.DPDispatcherSubmission')

        self.mock_machine = self.patcher_machine.start()
        self.mock_resources = self.patcher_resources.start()
        self.mock_task = self.patcher_task.start()
        self.mock_submission = self.patcher_submission.start()

        # Setup mock returns
        self.mock_machine.load_from_dict.return_value = Mock()
        self.mock_resources.load_from_dict.return_value = Mock()
        self.mock_submission.return_value.submission_hash = "test_hash"

        # Create temporary test directory
        self.test_dir = "test_temp_dir"
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        self.patcher_machine.stop()
        self.patcher_resources.stop()
        self.patcher_task.stop()
        self.patcher_submission.stop()

        # Clean up test directory
        if os.path.exists(self.test_dir):
            import shutil
            shutil.rmtree(self.test_dir)

    def test_init(self):
        """Test DpdispatcherExecutor initialization."""
        executor = DpdispatcherExecutor()
        
        self.assertEqual(executor.machine, self.mock_machine.load_from_dict.return_value)
        self.assertEqual(executor.resources, self.mock_resources.load_from_dict.return_value)
        
        self.mock_machine.load_from_dict.assert_called_once()
        self.mock_resources.load_from_dict.assert_called_once()

    def test_submit(self):
        """Test submit method for single job submission."""
        # Setup
        executor = DpdispatcherExecutor()
        job_dir = os.path.join(self.test_dir, "test_job")
        os.makedirs(job_dir, exist_ok=True)

        # Create test files
        Path(job_dir, "in.lammps").touch()
        Path(job_dir, "graph.pb").touch()

        # Execute
        submission_hash = executor.submit(job_dir)

        # Verify
        self.assertEqual(submission_hash, "test_hash")
        self.mock_submission.assert_called_once()
        self.mock_task.assert_called_once_with(
            command="lmp -i in.lammps",
            task_work_path="./",
            forward_files=["in.lammps", "*lmp", "graph.pb"],
            backward_files=["log.lammps", "dump.equi", "out.lmp"]
        )

    def test_group_submit(self):
        """Test group_submit method for multiple job submission."""
        # Setup
        executor = DpdispatcherExecutor()
        job_dir = os.path.join(self.test_dir, "test_group_job")
        
        # Create test directory structure
        task_dirs = ["task1", "task2", "task3"]
        for task in task_dirs:
            task_path = Path(job_dir) / task
            os.makedirs(task_path, exist_ok=True)
            Path(task_path, "in.lammps").touch()
        
        Path(job_dir, "graph.pb").touch()

        # Mock glob
        with patch('glob.glob') as mock_glob:
            mock_glob.return_value = [str(Path(job_dir) / task) for task in task_dirs]
            
            # Execute
            submission_hash = executor.group_submit(
                job_dir,
                subtasks_template="./task*",
                command="lmp -i in.lammps"
            )

            # Verify
            self.assertEqual(submission_hash, "test_hash")
            self.assertEqual(self.mock_task.call_count, len(task_dirs))
            self.mock_submission.assert_called_once()

    def test_group_submit_empty_directory(self):
        """Test group_submit method with empty directory."""
        # Setup
        executor = DpdispatcherExecutor()
        job_dir = os.path.join(self.test_dir, "empty_job")
        os.makedirs(job_dir, exist_ok=True)

        # Execute with empty directory
        with patch('glob.glob') as mock_glob:
            mock_glob.return_value = []
            submission_hash = executor.group_submit(job_dir)

            # Verify
            self.assertEqual(submission_hash, "test_hash")
            self.mock_submission.assert_called_once()
            self.assertEqual(self.mock_task.call_count, 0)

    def test_submit_invalid_path(self):
        """Test submit method with invalid path."""
        executor = DpdispatcherExecutor()
        invalid_paths = [
            "/non/existent/path",
            "",
            None
        ]
        
        for path in invalid_paths:
            with self.subTest(path=path):
                with self.assertRaises(Exception):  # You might want to be more specific about the exception
                    executor.submit(path)
        
    def test_submit_with_custom_command(self):
        """Test submit method with custom command."""
        executor = DpdispatcherExecutor()
        job_dir = os.path.join(self.test_dir, "test_job")
        os.makedirs(job_dir, exist_ok=True)

        Path(job_dir, "in.lammps").touch()
        Path(job_dir, "graph.pb").touch()

        custom_command = "lmp -pk gpu 1 -sf gpu -i in.lammps"
        submission_hash = executor.submit(job_dir, command=custom_command)

        self.assertEqual(submission_hash, "test_hash")
        self.mock_task.assert_called_once_with(
            command=custom_command,
            task_work_path="./",
            forward_files=["in.lammps", "*lmp", "graph.pb"],
            backward_files=["log.lammps", "dump.equi", "out.lmp"]
        )
    
    def test_submit_with_custom_command(self):
        """Test submit method with custom command."""
        executor = DpdispatcherExecutor()
        job_dir = os.path.join(self.test_dir, "test_job")
        os.makedirs(job_dir, exist_ok=True)

        Path(job_dir, "in.lammps").touch()
        Path(job_dir, "graph.pb").touch()

        custom_command = "lmp -pk gpu 1 -sf gpu -i in.lammps"
        submission_hash = executor.submit(job_dir, command=custom_command)

        self.assertEqual(submission_hash, "test_hash")
        self.mock_task.assert_called_once_with(
            command=custom_command,
            task_work_path="./",
            forward_files=["in.lammps", "*lmp", "graph.pb"],
            backward_files=["log.lammps", "dump.equi", "out.lmp"]
        )
    
    def test_submit_with_custom_command(self):
        """Test submit method with custom command."""
        executor = DpdispatcherExecutor()
        job_dir = os.path.join(self.test_dir, "test_job")
        os.makedirs(job_dir, exist_ok=True)

        Path(job_dir, "in.lammps").touch()
        Path(job_dir, "graph.pb").touch()

        custom_command = "lmp -pk gpu 1 -sf gpu -i in.lammps"
        submission_hash = executor.submit(job_dir, command=custom_command)

        self.assertEqual(submission_hash, "test_hash")
        self.mock_task.assert_called_once_with(
            command=custom_command,
            task_work_path="./",
            forward_files=["in.lammps", "*lmp", "graph.pb"],
            backward_files=["log.lammps", "dump.equi", "out.lmp"]
        )
    



if __name__ == '__main__':
    unittest.main()