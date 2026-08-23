import unittest

from dpti.lib.RemoteJob import JobStatus


class TestRemoteJobStatus(unittest.TestCase):
    def test_unknown_status_is_defined(self):
        """Scheduler fallbacks return a valid enum member for unknown states."""
        self.assertIs(JobStatus.unknown, JobStatus.unknow)


if __name__ == "__main__":
    unittest.main()
