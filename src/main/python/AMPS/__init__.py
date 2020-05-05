# import AMPSexperiment data structure
from .AMPSexperiment import AMPSexperiment, explore_background, return_signs
from .solvers import solve_AMPS
from pandas import read_csv
import pickle

from . import __path__
from pathlib import Path

__AMPS_PATH__ = Path(__file__).parent

with open(__AMPS_PATH__ / "SHGlookuptable.pickle", "rb") as fhd:
    _shglut = pickle.load(fhd)

with open(__AMPS_PATH__ / "TPFlookuptable.pickle", "rb") as fhd:
    _tpflut = pickle.load(fhd)


_AMPSboundaries = read_csv(__AMPS_PATH__ / "solution_boundaries.csv")
