import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import dpti
import dpti.equi, dpti.gdi, dpti.hti, dpti.hti_liq, dpti.hti_ice, dpti.hti_water, dpti.ti, dpti.ti_water

__all__ = ["dpti"]
