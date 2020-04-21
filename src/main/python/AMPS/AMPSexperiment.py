from PyBiodesy.DataStructures import correct_SHG
from PyBiodesy.Fitting import PhaseDetermination
from AMPS.solvers import compute_angles
from numpy import (
    sqrt,
    sin,
    rad2deg,
    deg2rad,
    cos,
    ones_like,
    array,
    linspace,
    meshgrid,
)


class AMPSexperiment:
    def __init__(self, dataframe, name, experiment=None):
        self.name = name
        self.data = dataframe
        self.unlabeled = dataframe[dataframe["frac_labeled"] == 0].copy()
        self.labeled = dataframe[dataframe["frac_labeled"] > 0].copy()
        self.grpdf = dataframe.groupby("frac_labeled").agg(["mean", "std"])
        self.labeled_index = self.labeled.index
        self.experiment = experiment
        self.p_inflection = None
        self.s_inflection = None
        self.pshg_bg = self.unlabeled["P-SHG"].mean()
        self.sshg_bg = self.unlabeled["S-SHG"].mean()
        self.Pphases = None
        self.Sphases = None

    def recalculate_stats(self):
        self.grpdf = self.data.groupby("frac_labeled").agg(["mean", "std"])

    def _zero_shift_fluorescence(self):
        # corrects over-subtracted fluorescence values
        unlabeled_P = self.unlabeled["P-FLcorr"].mean()
        unlabeled_S = self.unlabeled["S-FLcorr"].mean()

        self.data["P-FLcorr"] -= unlabeled_P
        self.data["S-FLcorr"] -= unlabeled_S

        self.unlabeled = self.data[self.data["frac_labeled"] == 0].copy()
        self.labeled = self.data[self.data["frac_labeled"] > 0].copy()

    def __str__(self):
        strout = f"AMPS experiment ({self.name})"
        return strout

    def __repr__(self):
        return self.__str__()

    @property
    def x_P(self):
        return self.data["P-FLcorr"].values

    @property
    def y_P(self):
        return self.data["P-SHG"].values

    @property
    def x_S(self):
        return self.data["S-FLcorr"].values

    @property
    def y_S(self):
        return self.data["S-SHG"].values

    @property
    def mean_x_P(self):
        return self.grpdf[("P-FLcorr", "mean")].values

    @property
    def mean_y_P(self):
        return self.grpdf[("P-SHG", "mean")].values

    @property
    def std_x_P(self):
        return self.grpdf[("P-FLcorr", "std")].values

    @property
    def std_y_P(self):
        return self.grpdf[("P-SHG", "std")].values

    @property
    def mean_x_S(self):
        return self.grpdf[("S-FLcorr", "mean")].values

    @property
    def mean_y_S(self):
        return self.grpdf[("S-SHG", "mean")].values

    @property
    def std_x_S(self):
        return self.grpdf[("S-FLcorr", "std")].values

    @property
    def std_y_S(self):
        return self.grpdf[("S-SHG", "std")].values

    def fit_phases(self):

        self.Pphases = PhaseDetermination(self.x_P, self.y_P)
        self.Sphases = PhaseDetermination(self.x_S, self.y_S)

        P_bg_est = self.unlabeled["P-SHG"].mean()
        S_bg_est = self.unlabeled["S-SHG"].mean()

        self.Pphases.run([P_bg_est, 1e-3, 1.047], fix="background")
        self.Sphases.run([S_bg_est, 1e-3, 1.047], fix="background")

        print("P-polarizaton : ", self.Pphases.optres.message)
        print("S-polarization : ", self.Sphases.optres.message)

        self.pshg_bg = self.Pphases.optres.params["bg"].value
        self.sshg_bg = self.Sphases.optres.params["bg"].value

        # plot using self.Pphases.fit.x vs self.Pphases.fit.y
        self.pshg_phase = self.Pphases.optres.params["delphi"].value
        self.sshg_phase = self.Sphases.optres.params["delphi"].value

        if rad2deg(self.pshg_phase) > 90.0:
            c1 = self.Pphases.optres.params["c1"].value
            # destructive interference
            # find the inflection point for fluorescence-vs-shg curve
            self.p_inflection = -(
                sqrt(self.pshg_bg) * sqrt(c1) * cos(self.pshg_phase) / c1
            )

        if rad2deg(self.sshg_phase) > 90.0:
            c1 = self.Sphases.optres.params["c1"].value
            self.s_inflection = -(
                sqrt(self.sshg_bg) * sqrt(c1) * cos(self.sshg_phase) / c1
            )

    def apply_SHG_correction(
        self,
        force_phase_P=None,
        force_phase_S=None,
        force_PSHG_bg=None,
        force_SSHG_bg=None,
        units="degree",
    ):
        """ correct SHG signal uses given phases in radians """

        if force_phase_P is None:
            phase_P = self.pshg_phase

        if force_phase_S is None:
            phase_S = self.sshg_phase

        x_P = self.labeled["P-FLcorr"].values
        x_S = self.labeled["S-FLcorr"].values

        if self.p_inflection is not None:
            # must be destructive interference
            p_signs = return_signs(x_P, self.p_inflection)
        else:
            p_signs = -ones_like(x_P)

        if self.s_inflection is not None:
            s_signs = return_signs(x_S, self.s_inflection)
        else:
            s_signs = -ones_like(x_S)

        if force_phase_P is not None:
            if units == "degree":
                phase_P = deg2rad(force_phase_P)
                if force_phase_P < 90.0:
                    p_signs = -ones_like(x_P)
                elif force_phase_P > 90.0:
                    p_signs = ones_like(x_P)

            elif units == "radians":
                phase_P = force_phase_P
                if rad2deg(phase_P) < 90.0:
                    p_signs = -ones_like(x_P)
                elif rad2deg(phase_P) > 90.0:
                    p_signs = ones_like(x_P)

        if force_phase_S is not None:
            if units == "degree":
                phase_S = deg2rad(force_phase_S)
                if force_phase_S < 90.0:
                    s_signs = -ones_like(x_S)
                elif force_phase_S > 90.0:
                    s_signs = ones_like(x_S)

            elif units == "radians":
                phase_S = force_phase_S
                if rad2deg(phase_P) < 90.0:
                    s_signs = -ones_like(x_S)
                elif rad2deg(phase_P) > 90.0:
                    s_signs = ones_like(x_S)

        _pshg_bg = force_PSHG_bg if force_PSHG_bg is not None else self.pshg_bg
        _sshg_bg = force_SSHG_bg if force_SSHG_bg is not None else self.sshg_bg

        if _pshg_bg is None:
            print("Please specify P-SHG background")
            return None
        if _sshg_bg is None:
            print("Please specify S-SHG background")
            return None

        correct_SHG(
            self.labeled,
            _pshg_bg,
            _sshg_bg,
            phase_P,
            phase_S,
            p_signs,
            s_signs,
        )

        self.data.loc[self.labeled_index, "P-SHGcorr"] = self.labeled[
            "P-SHGcorr"
        ]
        self.data.loc[self.labeled_index, "S-SHGcorr"] = self.labeled[
            "S-SHGcorr"
        ]

        self.recalculate_stats()

    def compute_TPFratio(self):

        self.labeled["TPFratio"] = (
            self.labeled["P-FLcorr"] / self.labeled["S-FLcorr"]
        )

        self.data.loc[self.labeled_index, "TPFratio"] = self.labeled["TPFratio"]
        self.recalculate_stats()

    def compute_SHGratio(self):

        self.labeled["SHGratio"] = (
            self.labeled["P-SHGcorr"] / self.labeled["S-SHGcorr"]
        )

        # assign back to original data
        self.data.loc[self.labeled_index, "SHGratio"] = self.labeled["SHGratio"]
        self.recalculate_stats()

    def solve_angles(self, silent=True, mode="fast"):

        if ("TPFratio" in self.data.columns) and (
            "SHGratio" in self.data.columns
        ):

            self.data[["angle", "distribution"]] = self.data.apply(
                compute_angles, axis=1, silent=silent, mode=mode
            )

            self.recalculate_stats()

        else:
            print(
                "You need to compute TPF and SHG ratios before solving for angles."
            )


def return_signs(vec, inflection):
    vecsign = [-1 if v > inflection else -1 for v in vec]
    return array(vecsign)


def explore_background(phases_min, phases_max, bg_min, bg_max, signal_obs):

    phasevec_deg = linspace(phases_min, phases_max, num=50)
    phasevec = deg2rad(phasevec_deg)

    bgvec = linspace(bg_min, bg_max, num=50)

    PHASES, BG = meshgrid(phasevec, bgvec, indexing="ij")

    arg = signal_obs / BG - sin(PHASES) ** 2

    # values that are negative 'arg' will yield imaginary sqrt

    return phasevec_deg, bgvec, arg
