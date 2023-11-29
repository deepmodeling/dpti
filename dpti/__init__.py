import sys
import os

NAME = "dpti"
SHORT_CMD = "dpti"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dpti
from . import lib
from . import equi
from . import hti
from . import hti_liq
from . import ti
from . import gdi