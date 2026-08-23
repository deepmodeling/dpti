import unittest
from enum import Enum

from dpti.dags.utils import is_transient_dag_run_state


class ExampleState(Enum):
    QUEUED = "queued"


class TestDagRunStates(unittest.TestCase):
    def test_healthy_transient_states_are_polled(self):
        """Queued, scheduled, running, and not-yet-visible runs are not failures."""
        for state in (None, "queued", "scheduled", "running", ExampleState.QUEUED):
            with self.subTest(state=state):
                self.assertTrue(is_transient_dag_run_state(state))

    def test_terminal_failure_is_not_transient(self):
        self.assertFalse(is_transient_dag_run_state("failed"))


if __name__ == "__main__":
    unittest.main()
