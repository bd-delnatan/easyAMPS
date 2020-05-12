"""
functions.py

This module contains routines that will be compiled on-the-fly for evaluating
integrals needed for SHG & TPF models.

"""

from numba import cfunc, jit, vectorize
from numba.core.types import intc, CPointer, float64
import numpy as np
from scipy.integrate import quad
from scipy import LowLevelCallable


def jit_integrand_function(integrand_function):
    """ decorator function to compile numerical integration routine

    This function is used to build AMPS functions as pre-compiled routines
    for faster evalution.

    From the StackOverflow answer:
    https://stackoverflow.com/questions/49683653/how-to-pass-additional-parameters-to-numba-cfunc-passed-as-lowlevelcallable-to-s

    We are using the function signature:
    double func(int n, double *xx)

    Where n is the length of xx array, xx[0] is the variable to be integrated over.
    The rest is the extra arguments that would normally be passed in 'args' of
    integrator 'scipy.integrate.quad'

    Usage:
    @jit_integrand_function
    def f(x, *args):
        a = args[0]
        b = args[1]
        return np.exp(-a*x/b)

    quad(f, 0, pi, args=(a,b))

    """
    jitted_function = jit(integrand_function, nopython=True)

    @cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):
        """ """
        return jitted_function(xx[0], xx[1], xx[2], xx[3])

    return LowLevelCallable(wrapped.ctypes)


@jit_integrand_function
def wrapped_gaussian(θ, *args):
    """ Sine-weighted wrapped gaussian function

    args needs to be [μ,σ,n] because we are putting them in an array of
    'doubles' to be referred to by '*xx' of the integrand function signature

    Args:
        θ (float): parameter to be integrated over.
        args[0] (float): location parameter, in radians.
        args[1] (float): scale parameter, in radians.
        args[2] (float): number of wrapped gaussians. 2 is good for up to 70-degree scale.

    """
    μ = args[0]
    σ = args[1]
    n = args[2]
    x1 = 2.0 * np.pi * np.arange(-n, n + 1.0) + θ
    x2 = 2.0 * np.pi * np.arange(-n, n + 1.0) - θ
    arg1 = (x1 - μ) / σ
    arg2 = (x2 - μ) / σ
    return (
        np.sum(np.exp(-0.5 * arg1 * arg1)) + np.sum(np.exp(-0.5 * arg2 * arg2))
    ) * np.sin(θ)


@jit_integrand_function
def p_shg_integrand(θ, *args):
    μ = args[0]
    σ = args[1]
    n = args[2]
    x1 = 2.0 * np.pi * np.arange(-n, n + 1.0) + θ
    x2 = 2.0 * np.pi * np.arange(-n, n + 1.0) - θ
    arg1 = (x1 - μ) / σ
    arg2 = (x2 - μ) / σ

    return (
        np.cos(θ) ** 3
        * (
            np.sum(np.exp(-0.5 * arg1 * arg1))
            + np.sum(np.exp(-0.5 * arg2 * arg2))
        )
        * np.sin(θ)
    )


@jit_integrand_function
def s_shg_integrand(θ, *args):
    μ = args[0]
    σ = args[1]
    n = args[2]
    x1 = 2.0 * np.pi * np.arange(-n, n + 1.0) + θ
    x2 = 2.0 * np.pi * np.arange(-n, n + 1.0) - θ
    arg1 = (x1 - μ) / σ
    arg2 = (x2 - μ) / σ

    return (
        np.sin(θ) ** 2
        * np.cos(θ)
        * (
            np.sum(np.exp(-0.5 * arg1 * arg1))
            + np.sum(np.exp(-0.5 * arg2 * arg2))
        )
        * np.sin(θ)
    )


@jit_integrand_function
def p_tpf_integrand(θ, *args):
    μ = args[0]
    σ = args[1]
    n = args[2]
    x1 = 2.0 * np.pi * np.arange(-n, n + 1.0) + θ
    x2 = 2.0 * np.pi * np.arange(-n, n + 1.0) - θ
    arg1 = (x1 - μ) / σ
    arg2 = (x2 - μ) / σ

    return (
        np.cos(θ) ** 4
        * np.sin(θ) ** 2
        * (
            np.sum(np.exp(-0.5 * arg1 * arg1))
            + np.sum(np.exp(-0.5 * arg2 * arg2))
        )
        * np.sin(θ)
    )


@jit_integrand_function
def s_tpf_integrand(θ, *args):
    μ = args[0]
    σ = args[1]
    n = args[2]
    x1 = 2.0 * np.pi * np.arange(-n, n + 1.0) + θ
    x2 = 2.0 * np.pi * np.arange(-n, n + 1.0) - θ
    arg1 = (x1 - μ) / σ
    arg2 = (x2 - μ) / σ

    return (
        np.sin(θ) ** 6
        * (
            np.sum(np.exp(-0.5 * arg1 * arg1))
            + np.sum(np.exp(-0.5 * arg2 * arg2))
        )
        * np.sin(θ)
    )


# cheap integration routine for approximating solutions
@vectorize([float64(float64, float64)])
def rough_shgratio(μ, σ):
    θ = np.linspace(0.0, np.pi, 100)
    W = np.zeros_like(θ)

    for n in range(-2, 3):
        x1 = 2.0 * np.pi * n + θ
        x2 = 2.0 * np.pi * n - θ
        arg1 = (x1 - μ) / σ
        arg2 = (x2 - μ) / σ
        W += np.exp(-0.5 * arg1 * arg1) + np.exp(-0.5 * arg2 * arg2)

    _P = np.sum(np.cos(θ) ** 3 * W * np.sin(θ))
    _S = np.sum(np.sin(θ) ** 2 * np.cos(θ) * W * np.sin(θ))

    return 1.88 * 4.0 * (_P / _S) ** 2


@vectorize([float64(float64, float64)])
def rough_tpfratio(μ, σ):
    θ = np.linspace(0.0, np.pi, 100)
    W = np.zeros_like(θ)

    for n in range(-2, 3):
        x1 = 2.0 * np.pi * n + θ
        x2 = 2.0 * np.pi * n - θ
        arg1 = (x1 - μ) / σ
        arg2 = (x2 - μ) / σ
        W += np.exp(-0.5 * arg1 * arg1) + np.exp(-0.5 * arg2 * arg2)

    _P = np.sum(np.cos(θ) ** 4 * np.sin(θ) ** 2 * W * np.sin(θ))
    _S = np.sum(np.sin(θ) ** 6 * W * np.sin(θ))

    return 1.88 * (8.0 * _P) / (3.0 * _S)


def nP_SHG(μ, σ, n=2.0):
    normalizer = quad(wrapped_gaussian, 0, np.pi, args=(μ, σ, n))[0]
    integrand = quad(p_shg_integrand, 0, np.pi, args=(μ, σ, n))[0]
    return (integrand / normalizer) ** 2


def nS_SHG(μ, σ, n=2.0):
    normalizer = quad(wrapped_gaussian, 0, np.pi, args=(μ, σ, n))[0]
    integrand = quad(s_shg_integrand, 0, np.pi, args=(μ, σ, n))[0]
    return (integrand / normalizer) ** 2 / 4.0


def nP_TPF(μ, σ, n=2.0):
    normalizer = quad(wrapped_gaussian, 0, np.pi, args=(μ, σ, n))[0]
    integrand = quad(p_tpf_integrand, 0, np.pi, args=(μ, σ, n))[0]
    return integrand / normalizer


def nS_TPF(μ, σ, n=2.0):
    normalizer = quad(wrapped_gaussian, 0, np.pi, args=(μ, σ, n))[0]
    integrand = quad(s_tpf_integrand, 0, np.pi, args=(μ, σ, n))[0]
    return (integrand / normalizer) * 3.0 / 8.0


def nSHGratio(μ, σ, fresnel_ratio=1.88, n=2.0, unit="radians"):
    if unit == "degree":
        μ *= np.pi / 180.0
        σ *= np.pi / 180.0
    P_integrand = quad(p_shg_integrand, 0, np.pi, args=(μ, σ, n))[0]
    S_integrand = quad(s_shg_integrand, 0, np.pi, args=(μ, σ, n))[0]
    return fresnel_ratio * 4.0 * (P_integrand / S_integrand) ** 2


def nTPFratio(μ, σ, fresnel_ratio=1.88, n=2.0, unit="radians"):
    if unit == "degree":
        μ *= np.pi / 180.0
        σ *= np.pi / 180.0
    P_integrand = quad(p_tpf_integrand, 0, np.pi, args=(μ, σ, n))[0]
    S_integrand = quad(s_tpf_integrand, 0, np.pi, args=(μ, σ, n))[0]
    return fresnel_ratio * (8.0 * P_integrand) / (3.0 * S_integrand)


def dP_SHG(μ, σ, n=2):
    h = 1e-5
    f = nP_SHG
    # wrt μ
    df_dμ = (f(μ + h, σ, n) - f(μ - h, σ, n)) / (2 * h)
    # wrt σ
    df_dσ = (f(μ, σ + h, n) - f(μ, σ - h, n)) / (2 * h)
    return df_dμ, df_dσ


def dS_SHG(μ, σ, n=2):
    h = 1e-5
    f = nS_SHG
    # wrt μ
    df_dμ = (f(μ + h, σ, n) - f(μ - h, σ, n)) / (2 * h)
    # wrt σ
    df_dσ = (f(μ, σ + h, n) - f(μ, σ - h, n)) / (2 * h)
    return df_dμ, df_dσ


def dP_TPF(μ, σ, n=2):
    h = 1e-5
    f = nP_TPF
    # wrt μ
    df_dμ = (f(μ + h, σ, n) - f(μ - h, σ, n)) / (2 * h)
    # wrt σ
    df_dσ = (f(μ, σ + h, n) - f(μ, σ - h, n)) / (2 * h)
    return df_dμ, df_dσ


def dS_TPF(μ, σ, n=2):
    h = 1e-5
    f = nS_TPF
    # wrt μ
    df_dμ = (f(μ + h, σ, n) - f(μ - h, σ, n)) / (2 * h)
    # wrt σ
    df_dσ = (f(μ, σ + h, n) - f(μ, σ - h, n)) / (2 * h)
    return df_dμ, df_dσ


def dSHGratio(μ, σ, n=2):
    h = 1e-5
    f = nSHGratio
    # wrt μ
    df_dμ = (f(μ + h, σ, n) - f(μ - h, σ, n)) / (2 * h)
    # wrt σ
    df_dσ = (f(μ, σ + h, n) - f(μ, σ - h, n)) / (2 * h)
    return df_dμ, df_dσ


def dTPFratio(μ, σ, n=2):

    h = 1e-5
    f = nTPFratio
    # wrt μ
    df_dμ = (f(μ + h, σ, n) - f(μ - h, σ, n)) / (2 * h)
    # wrt σ
    df_dσ = (f(μ, σ + h, n) - f(μ, σ - h, n)) / (2 * h)
    return df_dμ, df_dσ


# convenient function for generating ranges in radians given arguments in degree
def linspace_rad(start, stop, **kwargs):
    return np.linspace(np.deg2rad(start), np.deg2rad(stop), **kwargs)


# vectorized versions of the function
_ptpf = np.vectorize(nP_TPF, excluded=["n"])
_stpf = np.vectorize(nS_TPF, excluded=["n"])
_pshg = np.vectorize(nP_TPF, excluded=["n"])
_sshg = np.vectorize(nP_TPF, excluded=["n"])
_tpfratio = np.vectorize(nTPFratio, excluded=["fresnel_ratio", "n", "unit"])
_shgratio = np.vectorize(nSHGratio, excluded=["fresnel_ratio", "n", "unit"])
_tpfratio_der = np.vectorize(dTPFratio, excluded=["n"])
_shgratio_der = np.vectorize(dSHGratio, excluded=["n"])
