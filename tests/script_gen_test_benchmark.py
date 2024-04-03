#!/usr/bin/env python3
import json
from unittest.mock import patch

import numpy
from context import dpti

# assert os.path.isdir('equi_benchmark') is False

with patch.object(numpy.random, "randint", return_value=7858) as mock_method:
    # @patch('numpy.random')
    # def gen_equi_benchmark(patch_random):
    # patch_random.randint = MagicMock(return_value=7858)
    with open("npt.json") as f:
        equi_npt_benchmark_settings = json.load(f)
    dpti.equi.make_task(
        iter_name="equi_benchmark/npt", jdata=equi_npt_benchmark_settings
    )

    with open("npt_meam.json") as f:
        equi_npt_meam_benchmark_settings = json.load(f)
    dpti.equi.make_task(
        iter_name="equi_benchmark/npt_meam", jdata=equi_npt_meam_benchmark_settings
    )

# gen_equi_benchmark()
