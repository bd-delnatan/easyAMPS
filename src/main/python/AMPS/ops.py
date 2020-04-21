"""
ops.py

contains custom theano ops for use in PyMC3


"""
from .functions import (
    nP_SHG,
    nS_SHG,
    nP_TPF,
    nS_TPF,
    dP_SHG,
    dS_SHG,
    dP_TPF,
    dS_TPF,
)
import theano.tensor as T
import theano
import numpy as np


# Using the spline objects are simple
# spline(x,y,grid=False) returns a scalar

# for partial derivatives
# spline(x,y,dx=1,grid=False) df/dx
# spline(x,y,dy=1,grid=False) df/dy


# define convenient wrapper functions
def canonize_spline(spline):
    def _f(loc, scale):
        return spline(loc, scale, grid=False)

    return _f


def canonize_spline_deriv(spline):
    def _f(loc, scale):
        dloc = spline(loc, scale, dx=1, grid=False)
        dscale = spline(loc, scale, dy=1, grid=False)
        return dloc, dscale

    return _f


# define a custom theano op for the gradient for each op!
class NumericOp(theano.Op):
    itypes = [T.dscalar, T.dscalar]
    otypes = [T.dscalar]

    def __init__(self, expr, dexpr):
        self._expression = expr
        self._diffOp = NumericDiffOp(dexpr)

    def perform(self, node, inputs, outputs):
        location, scale, = inputs
        out = self._expression(location, scale)
        outputs[0][0] = np.array(out)

    def grad(self, inputs, grads):
        location, scale, = inputs
        dloc, dscale = self._diffOp(location, scale)
        return [grads[0] * dloc, grads[0] * dscale]


class NumericDiffOp(theano.Op):
    itypes = [T.dscalar, T.dscalar]
    otypes = [T.dscalar, T.dscalar]

    def __init__(self, expr):
        self._expression = expr

    def perform(self, node, inputs, outputs):
        location, scale, = inputs
        dloc, dscale = self._expression(location, scale)
        outputs[0][0] = np.array(dloc)
        outputs[1][0] = np.array(dscale)


# theano Ops using integration
theano_P_SHG = NumericOp(nP_SHG, dP_SHG)
theano_S_SHG = NumericOp(nS_SHG, dS_SHG)

theano_P_TPF = NumericOp(nP_TPF, dP_TPF)
theano_S_TPF = NumericOp(nS_TPF, dS_TPF)

# spline approximations
# sp_p_shg = canonize_spline(pshg_spline)
# sp_s_shg = canonize_spline(sshg_spline)
# sp_p_tpf = canonize_spline(ptpf_spline)
# sp_s_tpf = canonize_spline(stpf_spline)

# spder_p_shg = canonize_spline_deriv(pshg_spline)
# spder_s_shg = canonize_spline_deriv(sshg_spline)
# spder_p_tpf = canonize_spline_deriv(ptpf_spline)
# spder_s_tpf = canonize_spline_deriv(stpf_spline)

# and theano Ops using these spline approximations
# appx_P_SHG = NumericOp(sp_p_shg, spder_p_shg)
# appx_S_SHG = NumericOp(sp_s_shg, spder_s_shg)
# appx_P_TPF = NumericOp(sp_p_tpf, spder_p_tpf)
# appx_S_TPF = NumericOp(sp_p_tpf, spder_s_tpf)

__all__ = [
    "theano_P_SHG",
    "theano_S_SHG",
    "theano_P_TPF",
    "theano_S_TPF",
]
