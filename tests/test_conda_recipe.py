import os
import unittest


class TestCondaRecipe(unittest.TestCase):
    def test_runtime_dependencies_match_packaged_modules(self):
        """The no-deps conda install explicitly supplies runtime requirements."""
        recipe = os.path.join("..", "conda", "dpti", "meta.yaml")
        with open(recipe) as fp:
            content = fp.read()

        for dependency in (
            "apache-airflow >=2.0",
            "scipy",
            "numpy",
            "pymbar",
            "dargs",
            "dpdispatcher >=0.3",
            "sqlalchemy >=1.4.28,<2.0",
        ):
            with self.subTest(dependency=dependency):
                self.assertIn(f"- {dependency}", content)
        self.assertIn("import dpti.ti; import dpti.dags.dp_ti_gdi", content)


if __name__ == "__main__":
    unittest.main()
