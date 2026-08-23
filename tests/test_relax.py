import unittest

from dpti import relax


class TestRelaxImports(unittest.TestCase):
    def test_package_module_imports(self):
        """The relax module exposes its public task generator when installed."""
        self.assertTrue(callable(relax.make_task))


if __name__ == "__main__":
    unittest.main()
