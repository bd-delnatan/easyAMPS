"""
a collection of finite difference algorithms in Python

function that is passed to dfridr should be compiled by numba

"""
import numpy as np


def dfridr(func, x, h):

    # Npars = len(x)
    con = 1.4
    con2 = con * con
    big = 1e20
    ntab = 10
    safe = 2.0

    # error threshold
    errt = big

    # output
    out = None
    err = None
    A = np.zeros((ntab, ntab))

    assert h > 0, "h must be > 0!"

    hh = h * 1.0
    A[0, 0] = (func(x + hh) - func(x - hh)) / (2 * hh)

    for i in range(1, ntab):
        hh /= con
        A[0, i] = (func(x + hh) - func(x - hh)) / (2 * hh)
        fac = con2
        for j in range(1, i + 1):
            A[j, i] = (A[j - 1, i] * fac - A[j - 1, i - 1]) / (fac - 1.0)
            fac = con2 * fac
            errt = max(
                abs(A[j, i] - A[j - 1, i]), abs(A[j, i] - A[j - 1, i - 1])
            )
            if err is None or errt <= err:
                err = errt
                out = A[j, i]

        if abs(A[i, i] - A[i - 1, i - 1]) >= safe * err:
            break

    return out, err


def approx_2deriv(f, xlist, h=1e-2):

    # fix one parameter
    df1 = lambda x: f(x, xlist[1])
    df2 = lambda x: f(xlist[0], x)

    dval1, err1 = dfridr(df1, xlist[0], h)
    dval2, err2 = dfridr(df2, xlist[1], h)

    return (dval1, dval2), (err1, err2)
