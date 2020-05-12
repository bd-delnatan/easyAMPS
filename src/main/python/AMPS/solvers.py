""" module for solving AMPS from ratios """

import warnings
from scipy.optimize import root
from .functions import (
    rough_shgratio,
    rough_tpfratio,
    nSHGratio,
    nTPFratio,
    dSHGratio,
    dTPFratio,
)
import numpy as np
from numba import njit
from pandas import Series


def objective(p, rshg_obs, rtpf_obs):
    """ Function whose root we wish to find

    to be used with scipy.optimize.root

    Args:
        p: a tuple of mean angle and distribution [μ,σ], given in radians

    """
    μ, σ = p
    return [nSHGratio(μ, σ) - rshg_obs, nTPFratio(μ, σ) - rtpf_obs]


def jacobian(p, rshg_obs, rtpf_obs):

    μ, σ = p
    dSHG_dμ, dSHG_dσ = dSHGratio(μ, σ)
    dTPF_dμ, dTPF_dσ = dTPFratio(μ, σ)

    return [[dSHG_dμ, dSHG_dσ], [dTPF_dμ, dTPF_dσ]]


@njit(cache=True)
def approximate_solution(rshg_obs, rtpf_obs):
    # start with lower / upper bounds
    xvec = np.linspace(0.0349, 1.222, 75)
    _lb = np.zeros_like(xvec)
    _ub = np.ones_like(xvec) * 1.569

    # for TPF
    _lbsign_tpf = np.sign(rough_tpfratio(_lb, xvec) - rtpf_obs)
    _ubsign_tpf = np.sign(rough_tpfratio(_ub, xvec) - rtpf_obs)

    # for SHG
    _lbsign_shg = np.sign(rough_shgratio(_lb, xvec) - rshg_obs)
    _ubsign_shg = np.sign(rough_shgratio(_ub, xvec) - rshg_obs)

    # remove regions with no isocontours
    nosolution_tpf = _lbsign_tpf == _ubsign_tpf
    nosolution_shg = _lbsign_shg == _ubsign_shg

    # prepare variables for bisection search
    ly = _lb[~nosolution_tpf]
    uy = _ub[~nosolution_tpf]
    x_tpf = xvec[~nosolution_tpf]
    lsign = _lbsign_tpf[~nosolution_tpf]
    usign = _ubsign_tpf[~nosolution_tpf]

    # tolerance of roughly 0.2 degrees
    tol = 3.5e-3

    # do bisection search for tpf ratio
    while True:
        y_tpf = (ly + uy) / 2.0
        fmid = rough_tpfratio(y_tpf, x_tpf) - rtpf_obs
        msign = np.sign(fmid)
        ly[msign == lsign] = y_tpf[msign == lsign]
        uy[msign == usign] = y_tpf[msign == usign]
        if np.all(np.abs(uy - ly) < tol):
            # print(f"Converged after {it:d} iterations.")
            break

    # do bisection search for shg ratio
    ly = _lb[~nosolution_shg]
    uy = _ub[~nosolution_shg]
    x_shg = xvec[~nosolution_shg]
    lsign = _lbsign_shg[~nosolution_shg]
    usign = _ubsign_shg[~nosolution_shg]

    while True:
        y_shg = (ly + uy) / 2.0
        fmid = rough_shgratio(y_shg, x_shg) - rshg_obs
        msign = np.sign(fmid)
        ly[msign == lsign] = y_shg[msign == lsign]
        uy[msign == usign] = y_shg[msign == usign]
        if np.all(np.abs(uy - ly) < tol):
            # print(f"Converged after {it:d} iterations.")
            break

    # now find the rough intersectio between two segments
    # find common / shortest x-axis
    commonid = np.arange(min(y_shg.size, y_tpf.size))
    x_common = x_shg[commonid]
    y_shg = y_shg[commonid]
    y_tpf = y_tpf[commonid]

    # find index where y_shg is closest to y_tpf
    ydist = np.abs(y_shg - y_tpf)
    yminid = np.argmin(ydist)

    x_ans = x_common[yminid]
    avg_y = (y_shg[yminid] + y_tpf[yminid]) / 2.0

    sol = np.array([avg_y, x_ans])

    return sol


def solve_AMPS(rshg, rtpf, tolerance=1e-3, unit="degree", silent=True):
    """ returns AMPS solution

    Args:
        rshg (float): P/S SHG ratio, observed data.
        rtpf (float): P/S TPF ratio, observed data.
        tolerance (float): tolerance for root-finding algorithm
        unit (str): unit of the result. 'degree' or 'radians'

    Returns:
        angle, distribution

    """

    if silent:
        warnings.simplefilter("ignore")

    if np.isnan(rshg) or np.isnan(rtpf):
        if not silent:
            print(f"Skipping ... SHG-ratio {rshg:.3f}, TPF-ratio {rtpf:.3f}")
        return np.nan, np.nan

    try:

        x0 = approximate_solution(rshg, rtpf)
        # input arguments to objective is in radians
        sol = root(
            objective,
            x0,
            args=(rshg, rtpf),
            jac=jacobian,
            method="hybr",
            tol=tolerance,
        )

    except ZeroDivisionError:
        if not silent:
            print(
                f"Solution not found! at SHG-ratio {rshg:.3f}, TPF-ratio {rtpf:.3f}"
            )
        return np.nan, np.nan

    if sol.success:
        # since the solution is unbounded, we need to enforce it ourselves
        # if either of the solution is negative, it's wrong
        if sol.x[0] < 0 or sol.x[0] < 0:
            return np.nan, np.nan
        # or if it's beyond the Gaussian approximation model
        if sol.x[0] > 1.571 or sol.x[1] > 1.309:
            return np.nan, np.nan

        if unit == "degree":
            return np.rad2deg(sol.x[0]), np.rad2deg(sol.x[1])

        elif unit == "radians":
            return sol.x[0], sol.x[1]

    else:
        if not silent:
            print(
                f"Solution not found! at SHG-ratio {rshg:.3f}, TPF-ratio {rtpf:.3f}"
            )
        return np.nan, np.nan


def compute_angles(df, silent=True):
    """ to be used in DataFrame.apply()


    """
    shgratio = df["SHGratio"]
    tpfratio = df["TPFratio"]
    θ_μ, θ_σ = solve_AMPS(shgratio, tpfratio, silent=silent)
    return Series({"angle": θ_μ, "distribution": θ_σ})
