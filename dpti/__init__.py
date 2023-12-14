import os
import sys

NAME = "dpti"
SHORT_CMD = "dpti"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import dpti

from . import equi, gdi, hti, hti_liq, lib, ti
