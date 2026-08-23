#!/usr/bin/env bash
set -euo pipefail

# TI_taskflow is defined by workflow/DpFreeEnergy.py and must be loaded by Airflow.
example_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
config_file="${1:-${example_dir}/FreeEnergy.json}"
dag_id="${DPTI_AIRFLOW_DAG_ID:-TI_taskflow}"
work_base_dir="${DPTI_AIRFLOW_WORK_BASE_DIR:-${example_dir}}"

config="$(DPTI_AIRFLOW_WORK_BASE_DIR="${work_base_dir}" python -c '
import json
import os
import sys

with open(sys.argv[1]) as fp:
    data = json.load(fp)
data["work_base_dir"] = os.environ["DPTI_AIRFLOW_WORK_BASE_DIR"]
print(json.dumps(data))
' "${config_file}")"

airflow dags trigger "${dag_id}" --conf "${config}"
