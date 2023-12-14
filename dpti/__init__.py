import os
import sys

NAME = "dpti"
SHORT_CMD = "dpti"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

__all__ = [
    "dpti",
    "equi",
    "gdi",
    "hti",
    "hti_liq",
    "hti_ice",
    "hti_water",
    "ti",
    "ti_water",
    "lib",
]
