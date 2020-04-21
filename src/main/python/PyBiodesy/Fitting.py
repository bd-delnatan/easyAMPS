"""
This module contains all of the fitting-related functions and classes
commonly used in a typical workflow for analyzing SHG/TPF/biochemical data.

TO DO:
    - implement error propagation via autograd (from HIPS module?)

"""
import numpy as np
from string import Formatter
import lmfit
from collections import namedtuple


def logspace(xmin, xmax, num=100, fraction=0.5):
    """ Create a sequence of logarithmically-spaced vector

    Args:
        xmin(float): minimum number.
        xmax(float): maximum number.
        N(int): number of elements in output vector.
        fraction(float): fractional spacing between elements.

    Returns:
        ndarray

    """
    if xmax < xmin:
        raise ValueError("xmax must be larger than xmin.")

    if xmin > 0:
        # create a logarithmically-spaced vector between xmin to xmax
        b = np.log(xmin / xmax) / np.log(fraction)
        return xmax * fraction ** np.linspace(b, 0.0, num=num)
    else:
        raise ValueError(
            "Can't create a sequence with zero. "
            "Logarithm of zero does not exist"
        )


def geometric_mean(x):
    """ Computes geometric mean from input vector x """
    nonzero = x > 0
    lx = np.log(x[nonzero])
    return np.exp(lx.sum() / lx.size)


def deg2rad(x):
    if x is not None:
        return x * np.pi / 180.0
    elif x is None:
        return None


def rad2deg(x):
    if x is not None:
        return x * 180.0 / np.pi
    elif x is None:
        return None


# FITTING FUNCTIONS
def quadratic_function(p, x):
    # func(x) = a*x^2 + b*x + c
    a, b, c = p
    return a * x * x + b * x + c


class PartialFormatter(Formatter):
    def __init__(self, missing="~~", bad_fmt="!!"):
        self.missing, self.bad_fmt = missing, bad_fmt

    def get_field(self, field_name, args, kwargs):
        # Handle a key not found
        try:
            val = super(PartialFormatter, self).get_field(
                field_name, args, kwargs
            )
            # Python 3, 'super().get_field(field_name, args, kwargs)' works
        except (KeyError, AttributeError):
            val = None, field_name
        return val

    def format_field(self, value, spec):
        # handle an invalid format
        if value is None:
            return self.missing
        try:
            return super(PartialFormatter, self).format_field(value, spec)
        except ValueError:
            if self.bad_fmt is not None:
                return self.bad_fmt
            else:
                raise


class BindingCurve:
    """ A generic class for fitting and plotting binding curves

    Currently available binding models are :
        - Quadratic binding eqn., which accounts for ligand depletion. This is
        typically more accurate for tight-binding interactions in the low nano-
        molar regime.
        - Hyperbolic binding eqn., which is just a standard Langmuir isotherm.
        This is most commonly used for determining EC50 or equilibrium binding
        constant.

    This object has a @classmethod factory to instantiate the binding models
    listed above.

    Args:
        x(ndarray): ndarray of independent variables.
        y(ndarray): ndarray of independent variables. Must have the same shape
        as x.
        modality(str): either 'SHG' or 'TPF'. 'SHG' takes into account the
        quadratic nature of SHG data. N is proportional signal-squared.

    Usage:

        ::
        p = BindingCurve.Hyperbolic(x,y,sy=sy)
        p.fit()

    """

    @staticmethod
    def quadratic_binding(parameters, x, data=None, sigma=None, modality="SHG"):
        """ fitting function to be passed to lmfit.minimize

        if data is none, returns the calculated function.
        'sigma' is the standard deviation of data.
        'modality' is how the data was acquired.

        """
        dpars = parameters.valuesdict()
        Kd = dpars["Kd"]
        Rtot = dpars["Rtot"]
        offset = dpars["offset"]
        alpha = dpars["alpha"]

        c = Rtot + x + Kd
        Phi = (c - np.sqrt(c * c - 4 * Rtot * x)) / (2 * Rtot)

        if modality == "SHG":
            y_calc = 100.0 * ((Phi * alpha + 1.0) ** 2 - 1.0) + offset
        elif modality == "TPF":
            y_calc = offset * (Phi * alpha + 1.0)

        if data is None:
            return y_calc
        if sigma is None:
            return y_calc - data
        return (y_calc - data) / sigma

    @staticmethod
    def hyperbolic_binding(
        parameters, x, data=None, sigma=None, modality="SHG"
    ):
        """ fitting function to be passed to lmfit.minimize

        if data is none, returns the calculated function.
        'sigma' is the standard deviation of data.
        'modality' is how the data was acquired.

        """
        dpars = parameters.valuesdict()
        Kd = dpars["Kd"]
        offset = dpars["offset"]
        alpha = dpars["alpha"]

        Phi = x / (x + Kd)

        if modality == "SHG":
            y_calc = 100.0 * ((Phi * alpha + 1.0) ** 2 - 1.0) + offset
        elif modality == "TPF":
            y_calc = offset * (Phi * alpha + 1.0)

        if data is None:
            return y_calc
        if sigma is None:
            return y_calc - data
        return (y_calc - data) / sigma

    @staticmethod
    def langmuir_isotherm(parameters, x, data=None, sigma=None, modality="TPF"):
        dpars = parameters.valuesdict()
        Kd = dpars["Kd"]
        offset = dpars["offset"]
        Amplitude = dpars["Amplitude"]
        Phi = x / (x + Kd)

        if modality == "SHG":
            raise NotImplementedError(
                "This binding model does not have an SHG modality"
            )
        else:
            y_calc = Amplitude * Phi + offset

        if data is None:
            return y_calc
        if sigma is None:
            return y_calc - data
        return (y_calc - data) / sigma

    @staticmethod
    def cooperative_hyperbolic_binding(
        parameters, x, data=None, sigma=None, modality="SHG"
    ):
        """ fitting function to be passed to lmfit.minimize

        if data is none, returns the calculated function.
        'sigma' is the standard deviation of data.
        'modality' is how the data was acquired.

        """
        dpars = parameters.valuesdict()
        Kd = dpars["Kd"]
        offset = dpars["offset"]
        alpha = dpars["alpha"]
        n = dpars["n"]

        Phi = x ** n / (x ** n + Kd ** n)

        if modality == "SHG":
            y_calc = 100.0 * ((Phi * alpha + 1.0) ** 2 - 1.0) + offset
        elif modality == "TPF":
            y_calc = offset * (Phi * alpha + 1.0)

        if data is None:
            return y_calc
        if sigma is None:
            return y_calc - data
        return (y_calc - data) / sigma

    @staticmethod
    def cooperative_quadratic_binding(
        parameters, x, data=None, sigma=None, modality="SHG"
    ):
        """ fitting function to be passed to lmfit.minimize

        if data is none, returns the calculated function.
        'sigma' is the standard deviation of data.
        'modality' is how the data was acquired.

        """
        dpars = parameters.valuesdict()
        Kd = dpars["Kd"]
        Rtot = dpars["Rtot"]
        offset = dpars["offset"]
        alpha = dpars["alpha"]
        n = dpars["n"]

        c = Rtot + x ** n + Kd ** n
        Phi = (c - np.sqrt(c * c - 4 * Rtot * x ** n)) / (2 * Rtot)

        if modality == "SHG":
            y_calc = 100.0 * ((Phi * alpha + 1.0) ** 2 - 1.0) + offset
        elif modality == "TPF":
            y_calc = offset * (Phi * alpha + 1.0)

        if data is None:
            return y_calc
        if sigma is None:
            return y_calc - data
        return (y_calc - data) / sigma

    @classmethod
    def LigandDepletion(cls, *args, **kwargs):
        model_params = lmfit.Parameters()
        # these parameters should be positive
        model_params.add("Kd", value=1.0, min=0.0)
        model_params.add("Rtot", value=1.0, min=0.0)
        # these parameters are not constrained
        model_params.add("offset", value=1.0)
        model_params.add("alpha", value=1.0)

        return cls(
            BindingCurve.quadratic_binding, model_params, *args, **kwargs
        )

    @classmethod
    def ConventionalHyperbolic(cls, *args, **kwargs):
        """ hyperbolic binding curve with the conventional expression

        The binding curve for this class has three parameters:
        Amplitude, Kd and baseline offset.
                        x
            y = A * -------- + offset
                     x + Kd

        """

        model_params = lmfit.Parameters()
        model_params.add("Kd", value=1.0, min=0.0)
        model_params.add("Amplitude", value=1.0)
        model_params.add("offset", value=1.0)

        return cls(
            BindingCurve.langmuir_isotherm, model_params, *args, **kwargs
        )

    @classmethod
    def Hyperbolic(cls, *args, **kwargs):
        """ hyperbolic binding curve with "SHG"-derived expression.

        The binding curve for this class has three parameters:
        scaling parameter alpha, Kd and "offset". Here the offset and
        amplitude term has been combined.

            y = offset * ( phi * alpha + 1 )

        """
        model_params = lmfit.Parameters()
        # these parameters should be positive
        model_params.add("Kd", value=1.0, min=0.0)
        # these parameters are not constrained
        model_params.add("offset", value=1.0)
        model_params.add("alpha", value=1.0)

        return cls(
            BindingCurve.hyperbolic_binding, model_params, *args, **kwargs
        )

    @classmethod
    def CooperativeHyperbolic(cls, *args, **kwargs):
        """ cooperative hyperbolic binding curve, conventional expression

        The binding curve for this class has three parameters:
        scaling parameter alpha, Kd and baseline offset.

            y =

        """
        model_params = lmfit.Parameters()
        # these parameters should be positive
        model_params.add("Kd", value=1.0, min=0.0)
        # these parameters are not constrained
        model_params.add("offset", value=1.0)
        model_params.add("alpha", value=1.0)
        model_params.add("n", value=1.0, min=0.0)

        return cls(
            BindingCurve.cooperative_hyperbolic_binding,
            model_params,
            *args,
            **kwargs,
        )

    @classmethod
    def CooperativeLigandDepletion(cls, *args, **kwargs):
        model_params = lmfit.Parameters()
        # these parameters should be positive
        model_params.add("Kd", value=1.0, min=0.0)
        model_params.add("Rtot", value=1.0, min=0.0)
        # these parameters are not constrained
        model_params.add("offset", value=1.0)
        model_params.add("alpha", value=1.0)
        model_params.add("n", value=1.0, min=0.0)

        return cls(
            BindingCurve.cooperative_quadratic_binding,
            model_params,
            *args,
            **kwargs,
        )


    def __init__(
        self, binding_model, parameters, x, y=None, sy=None, modality="SHG"
    ):
        """ Generic template for binding curve models

        Instantiate this class with its binding_model and proper parameters.
        Implement the binding_model as a 'staticmethod'

        """
        x = np.array(x)

        if y is not None:
            y = np.array(y)
            # populate data for fitting
            mask = ~np.isnan(x) & ~np.isnan(y)
            self.y = y[mask]
        else:
            self.y = None
            mask = np.array(x.size, dtype=np.bool)
        self.x = x[mask]

        # can't fit more variables than parameters
        if len(self.x) <= len(parameters):
            self.valid = False
        else:
            self.valid = True

        self.sy = sy

        if self.sy is not None:
            self.weighted = True
            self.sy = np.array(sy)
        else:
            self.weighted = False

        self.modality = modality
        self.fcn = binding_model
        self.params = parameters
        self.optres = None

    def run(
        self, fit_method="leastsq", fixpars=None, use_previous=True, **fit_kws
    ):

        # initial parameters, if doesnt exist, use default
        if self.optres is None:
            init_pars = self.params.copy()
        else:
            # Otherwise, use previous fits. Useful for MCMC
            init_pars = self.optres.params.copy()

        # do "reasonable" initial guesses
        y_at_xmax = self.y[np.argmax(self.x)].mean()
        y_at_xmin = self.y[np.argmin(self.x)].mean()
        init_pars["Kd"].value = geometric_mean(self.x)
        init_pars["offset"].value = y_at_xmin

        if "alpha" in init_pars.keys():
            if self.modality == "TPF":
                init_pars["alpha"].value = (y_at_xmax - y_at_xmin) / y_at_xmin
            elif self.modality == "SHG":
                init_pars["alpha"].value = (
                    np.sqrt((y_at_xmax - y_at_xmin) / 100.0 + 1) - 1.0
                )

        if "Amplitude" in init_pars.keys():
            init_pars["Amplitude"].value = y_at_xmax - y_at_xmin

        # if 'fixpars' is passed, then the parameters will be fixed with given value
        if fixpars is not None:
            if isinstance(fixpars, dict):
                for par_name, par_value in fixpars.items():
                    if par_name in init_pars.keys():
                        init_pars[par_name].value = par_value
                        init_pars[par_name].vary = False
                    else:
                        raise KeyError(
                            "Parameter {:s} is not recognized.".format(par_name)
                        )
            elif fixpars == "all":
                for par_name in init_pars.keys():
                    init_pars[par_name].vary = False
            else:
                raise ValueError("fixpars argument is not recognized.")

        else:
            # if no parameters are fixed, check if using previous result
            if use_previous and self.optres is not None:
                init_pars = self.optres.params.copy()

        if self.valid:
            # if the model if valid, try to do the fitting
            try:
                if fit_method == "emcee":
                    # if using emcee, check if residual is weighted by known
                    # sigmas (passed to object as 'sy')
                    emceepars = {"is_weighted": self.weighted}
                    minpars = {**fit_kws, **emceepars}
                else:
                    minpars = fit_kws

                self.minimizer = lmfit.Minimizer(
                    userfcn=self.fcn,
                    params=init_pars,
                    fcn_args=(self.x,),
                    fcn_kws={
                        "data": self.y,
                        "sigma": self.sy,
                        "modality": self.modality,
                    },
                )

                self.optres = self.minimizer.minimize(
                    method=fit_method, **minpars
                )

            except ValueError:
                print("####### Oops something happened here. DEBUG #######")
                print("Initial parameters")
                for par_name, par in init_pars.items():
                    print(par_name, par.value)
                raise

            self.model = self._form_model(
                self.optres.params, modality=self.modality
            )

            # use self.model to form ideal/fitted curve
            self._form_ideal()

        else:
            # invalid model is not fitted, so all parameters are fixed
            # because we still need to run the minimizer to get result
            init_pars = self.params.copy()
            for par_name in init_pars.keys():
                init_pars[par_name].vary = False

            self.minimizer = lmfit.Minimizer(
                fcn=self.fcn,
                params=init_pars,
                fcn_args=(self.x,),
                fcn_kws={
                    "data": self.y,
                    "sigma": self.sy,
                    "modality": self.modality,
                },
            )

            self.optres = self.minimizer.minimize(method=fit_method, **fit_kws)

            for param_key in self.optres.params.keys():
                self.optres.params[param_key].value = np.nan

    def _form_model(self, pars=None, modality=None):
        if modality is None:
            modality = self.modality
        if pars is None:
            pars = self.params

        # the fitted model
        def _f(x):
            return self.fcn(pars, x, modality=modality)

        return _f

    def _form_ideal(self, xscale="log", Npts=100):

        Fit = namedtuple("Fit", ["x", "y"])

        if xscale == "linear":
            xsmooth = np.linspace(self.x.min(), self.x.max(), num=Npts)
        elif xscale == "log":
            xsmooth = logspace(self.x.min(), self.x.max(), num=Npts)

        self.fit = Fit(x=xsmooth, y=self.model(xsmooth))

    @property
    def Kd(self):
        if self.optres is not None:
            if self.optres.params['Kd'].value is not None:
                return self.optres.params['Kd'].value


class PhaseDetermination:
    """ Object for handling phase-determination

    This class handles fitting for phase-determination.

    Note:
        Angular units are given in radians.

    """

    def __init__(self, x, y):
        """ Phase-determination object

        Note:
            This object will treat each data point individually, no averaging is
            done

        Args:
            x (numpy vector): linearly-spaced x-axis from fluorescence channel
            data or a known mixture of labeled:unlabeled protein
            y (numpy vector): data from SHG channel

        """
        self.x = x
        self.y = y
        self.res = None
        self.res_str = None

    def f(self, p, x):
        bg, c1, delphi = p
        ycalc = (
            bg + 2 * np.sqrt(bg) * np.sqrt(c1) * np.cos(delphi) * x + c1 * x * x
        )
        return ycalc

    def resid(self, params, x, yobs):
        p = [params["bg"], params["c1"], params["delphi"]]
        return yobs - self.f(p, x)

    def run(self, p0, fix=None):
        """ do phase-determination fitting

        Args:
            p0 (list of floats): background, c1, phase difference in radians

        """
        self.initpars = lmfit.Parameters()

        self.initpars.add("c1", value=p0[1], min=0)

        if fix == "both":
            self.initpars.add("bg", value=p0[0], min=1e-4, vary=False)
            self.initpars.add(
                "delphi", value=p0[2], min=1e-4, max=np.pi, vary=False
            )
        elif fix == "background":
            self.initpars.add("bg", value=p0[0], min=1e-4, vary=False)
            self.initpars.add("delphi", value=p0[2], min=1e-4, max=np.pi)
        elif fix == "phase":
            self.initpars.add("bg", value=p0[0], min=1e-4)
            self.initpars.add(
                "delphi", value=p0[2], min=1e-4, max=np.pi, vary=False
            )
        else:
            self.initpars.add("bg", value=p0[0], min=0)
            self.initpars.add("delphi", value=p0[2], min=1e-4, max=np.pi)

        self.fitter = lmfit.Minimizer(
            self.resid, self.initpars, fcn_args=(self.x, self.y)
        )
        self.optres = self.fitter.minimize()
        # extract fitted parameters
        _popt = self.optres.params
        self.bg, self.bg_stderr = _popt["bg"].value, _popt["bg"].stderr
        self.c1, self.c1_stderr = _popt["c1"].value, _popt["c1"].stderr
        self.delphi = _popt["delphi"].value
        self.delphi_err = _popt["delphi"].stderr
        self.phase = rad2deg(_popt["delphi"].value)
        self.phase_stderr = rad2deg(_popt["delphi"].stderr)

        _str = (
            "SHG_B = {:.2f} +/- {:<.2f}\n"
            "C1 = {:.2e} +/- {:<.2e}\n"
            "Δφ = {:.2f}˚ +/- {:<.2f}˚\n"
            "in radians, {:.3f}  +/- {:.3f}"
        )

        fmt = PartialFormatter()

        self.res_str = fmt.format(
            _str,
            *[
                self.bg,
                self.bg_stderr,
                self.c1,
                self.c1_stderr,
                self.phase,
                self.phase_stderr,
                self.delphi,
                self.delphi_err,
            ],
        )

        self.model = lambda x: self.f([self.bg, self.c1, self.delphi], x)
        self.set_fit()

    def set_fit(self, npts=100):
        Fit = namedtuple("Fit", ["x", "y"])
        xsmooth = np.linspace(self.x.min(), self.x.max(), num=100)
        self.fit = Fit(x=xsmooth, y=self.model(xsmooth))
