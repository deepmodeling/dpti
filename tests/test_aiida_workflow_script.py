import ast
import os
import unittest


class TestAiidaWorkflowScript(unittest.TestCase):
    def test_workflow_execution_is_main_guarded(self):
        """Importing the script must not load a profile or start a workflow."""
        script = os.path.join("..", "workflow", "DpFreeEnergy-aiida.py")
        with open(script) as fp:
            tree = ast.parse(fp.read(), filename=script)

        top_level_calls = [
            node
            for node in tree.body
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        ]
        self.assertEqual(top_level_calls, [])
        self.assertTrue(any(isinstance(node, ast.If) for node in tree.body))


if __name__ == "__main__":
    unittest.main()
