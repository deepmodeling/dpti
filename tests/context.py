import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import dpti
import dpti.gdi  # noqa: F401

__all__ = ["dpti"]
