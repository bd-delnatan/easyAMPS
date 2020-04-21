"""
The 'integrals' module contains convenient functions for evaluating
numerical integrals to solve the SHG and TPF angular distributions.

"""

from scipy.special import i0
import numpy as np
from scipy.optimize import minimize

θvec = np.linspace(0.0, np.pi, num=500)
p0 = [50.0, 25.0]


def deg2rad(x):
    return x * (np.pi / 180.0)


def rad2deg(x):
    return x * (180.0 / np.pi)


def p_SHG_fun(x):
    return np.power(np.cos(x), 3)


def s_SHG_fun(x):
    return np.power(np.sin(x), 2) * np.cos(x)


def p_TPF_fun(x):
    return np.power(np.cos(x), 4) * np.power(np.sin(x), 2)


def s_TPF_fun(x):
    return np.power(np.sin(x), 6)


def f(θ, μ, σ):
    """ un-normalized Gaussian function

    Args:
        θ(1D numpy ndarray): a vector of points to evaluate the function
        μ(float): location parameter. The mean.
        σ(float): scale parameter. The standard deviation.

    Returns:
        a vector of the function values.

    """
    arg = (θ - μ) / σ
    return np.exp(-0.5 * (arg * arg))


def gaussian_pdf(θ, μ, σ):
    """ normalized Gaussian function

    Args:
        θ(1D numpy ndarray): a vector of points to evaluate the function
        μ(float): location parameter. The mean.
        σ(float): scale parameter. The standard deviation.

    Returns:
        a vector of the function values.

    """
    arg = (θ - μ) / σ
    return np.exp(-0.5 * (arg * arg)) / np.sqrt(2.0 * np.pi * σ ** 2)


def vonmises(θ, μ, σ):
    κ = 1 / (2 * σ ** 2)
    integrand = np.exp(κ * np.cos(θ - μ))
    return integrand / (2 * np.pi * i0(κ))


# the symmetric semi-circular Von Mises distribution, for θ in [0,π]
def semi_vonmises(θ, μ, σ):
    """ the planar-symmetric Von Mises distribution function

    This function wraps (reflects) with respect to the boundaries at 0 and
    180 degrees (or 0 to pi in radians). Used to evaluate expectation value
    in a forward model to compute SHG and TPF intensities.

    Args:
        θ(1D numpy ndarray): a vector of points to evaluate the function
        μ(float): location parameter. The mean.
        σ(float): scale parameter. The standard deviation.

    Returns:
        a vector of the function values.

    """
    κ = 1 / (2 * σ ** 2)
    integrand = np.exp(κ * np.cos(θ - μ)) + np.exp(κ * np.cos(θ + μ))
    return integrand / (2 * np.pi * i0(κ))


# the wrapped Gaussian (every 2π, with +/- π to wrap around limit 0->π)
def fp(θ, μ, σ, n=2):
    nlims = np.arange(n, -(n + 1), -1)
    # for the discrete increment shifted input angles for -n, -n-1, 0, n-1, n
    θplus = 2.0 * np.pi * nlims[np.newaxis, :] + θ[:, np.newaxis]
    θminus = 2.0 * np.pi * nlims[np.newaxis, :] - θ[:, np.newaxis]
    return f(θplus, μ, σ).sum(axis=1) + f(θminus, μ, σ).sum(axis=1)


# E is the expectation operator
def E(θ, μ, σ, f):
    """ compute the expectation of f(θ), <f(θ)> using a Gaussian

    The expectation is computed over a weighted, wrapped Gaussian distribution
    ranging from 0 to π angles.

    Args:
        θ(ndarray): a vector with the angular range in radians. This will be
        integrated over.
        μ(float): mean angular tilt
        σ(float): angular distribution width
        f(func): a Python function to evaluate f(θ)

    """
    dθ = θ[1] - θ[0]
    _fp = fp(θ, μ, σ, n=2) * np.sin(θ) * dθ
    _Z = np.sum(_fp)
    return np.sum(f(θ) * _fp) / _Z


def E_vm(θ, μ, σ, f):
    """ compute the expectation of f(θ), <f(θ)> using the VonMises distribution

    The expectation is computed over a weighted, semi-circular VonMises
    distribution with cylindrical symmetry, with angles ranging from 0 to π.

    Args:
        θ(ndarray): a vector with the angular range
        μ(float): mean angular tilt
        σ(float): angular distribution width
        f(func): a Python function to evaluate f(θ)

    """

    dθ = θ[1] - θ[0]
    _fp = semi_vonmises(θ, μ, σ) * np.sin(θ) * dθ
    _Z = np.sum(_fp)
    return np.sum(f(θ) * _fp) / _Z


def R_SHG(μ, σ, θvec, fresnel_ratio=1.88, distribution="Gaussian"):
    """ compute the SHG ratio for a given mean angle and distribution width

    This function uses the discrete approximation method. The discretization
    is specified by input vector θvec. There are two distributions implemented
    'Gaussian' and 'VonMises'.

    Args:
        μ(float) : mean tilt angle in degrees.
        σ(float) : angular distribution width in degrees.
        θvec(ndarray): a vector of angles in radians, from 0 to π.
        fresnel_ratio(float): the fresnel factor ratio, (fz/fy)^4. This is
            2.59 in the alpha system.
        distribution(str): the unimodal probability distribution. 'Gaussian'
        or 'VonMises'

    """
    E_op = {"Gaussian": E, "VonMises": E_vm}
    pA = E_op[distribution](θvec, deg2rad(μ), deg2rad(σ), p_SHG_fun)
    sA = E_op[distribution](θvec, deg2rad(μ), deg2rad(σ), s_SHG_fun)
    arg = (pA * pA) / (sA * sA)
    return 4.0 * fresnel_ratio * arg


def R_TPF(μ, σ, θvec, fresnel_ratio=1.88, distribution="VonMises"):
    """ compute the TPF ratio for a given mean angle and distribution width

    This function uses the discrete approximation method. The discretization
    is specified by input vector θvec. There are two distributions implemented
    'Gaussian' and 'VonMises'.

    Args:
        μ(float) : mean tilt angle in degrees.
        σ(float) : angular distribution width in degrees.
        θvec(ndarray): a vector of angles in radians, from 0 to π.
        fresnel_ratio(float): the fresnel factor ratio, (fz/fy)^4. This is
            2.59 in the alpha system.
        distribution(str): the unimodal probability distribution. 'Gaussian'
            or 'VonMises'

    """
    E_op = {"Gaussian": E, "VonMises": E_vm}
    pA = E_op[distribution](θvec, deg2rad(μ), deg2rad(σ), p_TPF_fun)
    sA = E_op[distribution](θvec, deg2rad(μ), deg2rad(σ), s_TPF_fun)
    arg = pA / sA
    return 8.0 * fresnel_ratio * arg / 3.0


def errorfunction(
    x0, Rshg_obs, Rtpf_obs, θvec, fresnel_ratio=1.88, distribution="Gaussian"
):
    """ Error function to pass to an optimizer for finding a Gaussian parameters

        To use pass this to scipy.optimize.minimize and place bounds on
        the parameters ::
            # points to do integration
            θvec = np.linspace(0.0, np.pi, num=400)
            # initial guess is 40 degrees angle with 20 degrees width
            p0 = [40.0, 20.0]
            SHG_ratio_data = 7.06
            TPF_ratio_data = 0.61

            # error tolerance can be adjusted also by passing 'tol'
            # you can also try the 'slsqp' method. It may be more robust
            # than bfgs. Here the boundaries are imposed on the solution.
            opts =  minimize(errorfunction, [40.0, 20.0],
                    args=(SHG_ratio_data, TPF_ratio_data, θvec,),
                    method='SLSQP',
                    bounds=((1e-3,179.9), (1e-3, 70.0)))

    Args:
        Rshg_obs(float): the measured P/S ratio from SHG data.
        Rtpf_obs(float): the measured P/S ratio from TPF data.
        θ(vector): a vector of angles from 0 to π to do the integration. Use
            at least 100 points ranging from 0.0 to π. This is in RADIANS!!!
        distribution(str): "Gaussian" or "VonMises".

    Returns:
        float: sum of squared error
    """
    μ, σ = x0
    Rshg_calc = R_SHG(
        μ, σ, θvec, fresnel_ratio=fresnel_ratio, distribution=distribution
    )
    Rtpf_calc = R_TPF(
        μ, σ, θvec, fresnel_ratio=fresnel_ratio, distribution=distribution
    )
    return (Rshg_obs - Rshg_calc) ** 2 + (Rtpf_obs - Rtpf_calc) ** 2


def solve_angle(shgratio, tpfratio, warnings=False):
    resopt = minimize(
        errorfunction,
        p0,
        args=(shgratio, tpfratio, θvec),
        method="SLSQP",
        bounds=((1e-3, 179.9), (1e-3, 70.0)),
    )

    if resopt.fun < 5e-4:
        angle, distribution = resopt.x[0], resopt.x[1]
    else:
        if warnings:
            print("No solutions found!")
        angle, distribution = np.nan, np.nan

    return angle, distribution
