# import AMPSexperiment data structure
from .AMPSexperiment import AMPSexperiment, explore_background, return_signs
from .solvers import solve_AMPS
import pickle
import pandas as pd
from . import __path__
from pathlib import Path

__AMPS_PATH__ = Path(__path__[0])

with open(__AMPS_PATH__ / "SHGlookuptable.pickle", "rb") as fhd:
    _shglut = pickle.load(fhd)

with open(__AMPS_PATH__ / "TPFlookuptable.pickle", "rb") as fhd:
    _tpflut = pickle.load(fhd)

_AMPSboundaries = pd.read_csv(__AMPS_PATH__ / "solution_boundaries.csv")
