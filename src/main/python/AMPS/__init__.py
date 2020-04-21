# import AMPSexperiment data structure
from .AMPSexperiment import AMPSexperiment, explore_background
from .solvers import solve_AMPS
import pickle
from . import __path__
from pathlib import Path

__AMPS_PATH__ = __path__[0]

with open(Path(__AMPS_PATH__) / "SHGlookuptable.pickle", "rb") as fhd:
    _shglut = pickle.load(fhd)

with open(Path(__AMPS_PATH__) / "TPFlookuptable.pickle", "rb") as fhd:
    _tpflut = pickle.load(fhd)
