""" module for solving AMPS from ratios """

import warnings
from scipy.optimize import root, minimize
from .functions import nSHGratio, nTPFratio
from numpy import rad2deg, nan, isnan
from pandas import Series


def objective(p, rshg_obs, rtpf_obs):
    """ Function whose root we wish to find

    to be used with scipy.optimize.root

    Args:
        p: a tuple of mean angle and distribution [μ,σ], given in radians

    """
    μ, σ = p
    return [rshg_obs - nSHGratio(μ, σ), rtpf_obs - nTPFratio(μ, σ)]


def objective2(x0, rshg_obs, rtpf_obs, fresnel_ratio=1.88):

    μ, σ = x0

    rshg_calc = nSHGratio(μ, σ)
    rtpf_calc = nTPFratio(μ, σ)

    return (rshg_obs - rshg_calc) ** 2 + (rtpf_obs - rtpf_calc) ** 2


def solve_cons_AMPS(rshg, rtpf, silent=True, unit="degree"):

    if isnan(rshg) or isnan(rtpf):
        if not silent:
            print(f"Skipping ... SHG-ratio {rshg:.3f}, TPF-ratio {rtpf:.3f}")
        return nan, nan

    resopt = minimize(
        objective2,
        [0.5, 0.43],
        args=(rshg, rtpf),
        method="SLSQP",
        bounds=((1e-3, 1.57), (3.5e-2, 1.484)),
    )

    if resopt.fun < 5e-4:
        angle, distribution = resopt.x[0], resopt.x[1]
        if unit == "degree":
            return rad2deg(angle), rad2deg(distribution)
        else:
            return angle, distribution
    else:
        if not silent:
            print(
                f"Solution not found! at SHG-ratio {rshg:.3f}, TPF-ratio {rtpf:.3f}"
            )
        return nan, nan


def solve_AMPS(rshg, rtpf, tolerance=1e-4, unit="degree", silent=True):
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

    if isnan(rshg) or isnan(rtpf):
        if not silent:
            print(f"Skipping ... SHG-ratio {rshg:.3f}, TPF-ratio {rtpf:.3f}")
        return nan, nan

    try:
        sol = root(objective, [0.5, 0.43], args=(rshg, rtpf), tol=tolerance)
    except ZeroDivisionError:
        if not silent:
            print(
                f"Solution not found! at SHG-ratio {rshg:.3f}, TPF-ratio {rtpf:.3f}"
            )
        return nan, nan

    if sol.success:
        # since the solution is unbounded, we need to enforce it ourselves
        # if either of the solution is negative, it's wrong
        if sol.x[0] < 0 or sol.x[1] < 0.0349:
            return nan, nan
        # or if it's beyond the Gaussian approximation model
        if sol.x[0] > 1.571 or sol.x[1] > 1.309:
            return nan, nan

        if unit == "degree":
            return rad2deg(sol.x[0]), rad2deg(sol.x[1])

        elif unit == "radians":
            return sol.x[0], sol.x[1]

    else:
        if not silent:
            print(
                f"Solution not found! at SHG-ratio {rshg:.3f}, TPF-ratio {rtpf:.3f}"
            )
        return nan, nan


def compute_angles(df, silent=True, mode="fast"):
    """ to be used in DataFrame.apply()


    """
    shgratio = df["SHGratio"]
    tpfratio = df["TPFratio"]

    if mode == "fast":
        θ_μ, θ_σ = solve_AMPS(shgratio, tpfratio, silent=silent)
    else:
        θ_μ, θ_σ = solve_cons_AMPS(shgratio, tpfratio, silent=silent)
    return Series({"angle": θ_μ, "distribution": θ_σ})
