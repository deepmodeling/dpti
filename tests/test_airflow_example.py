import json
import os
import stat
import subprocess
import tempfile
import unittest


class TestAirflowExample(unittest.TestCase):
    def test_script_resolves_checkout_paths_and_dag_name(self):
        """The example sends portable configuration to the documented DAG."""
        repository_root = os.path.abspath("..")
        script = os.path.join(repository_root, "examples", "airflow.sh")
        with tempfile.TemporaryDirectory() as tempdir:
            capture_path = os.path.join(tempdir, "args.txt")
            airflow = os.path.join(tempdir, "airflow")
            with open(airflow, "w") as fp:
                fp.write('#!/bin/sh\nprintf "%s\\n" "$@" > "$AIRFLOW_CAPTURE"\n')
            os.chmod(airflow, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            env = os.environ.copy()
            env["PATH"] = tempdir + os.pathsep + env["PATH"]
            env["AIRFLOW_CAPTURE"] = capture_path

            subprocess.run([script], cwd=repository_root, env=env, check=True)

            with open(capture_path) as fp:
                arguments = fp.read().splitlines()
        self.assertEqual(arguments[:3], ["dags", "trigger", "TI_taskflow"])
        config = json.loads(arguments[arguments.index("--conf") + 1])
        self.assertEqual(
            config["work_base_dir"], os.path.join(repository_root, "examples")
        )

    def test_json_contains_no_private_home_path(self):
        config_path = os.path.join("..", "examples", "FreeEnergy.json")
        with open(config_path) as fp:
            config = json.load(fp)

        self.assertEqual(config["work_base_dir"], ".")


if __name__ == "__main__":
    unittest.main()
