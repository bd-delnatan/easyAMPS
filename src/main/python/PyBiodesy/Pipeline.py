# CRC pipeline helpers
import pandas as pd
import pdb
from PyBiodesy.Fitting import BindingCurve, logspace
from PyBiodesy.integrals import (
    fp,
    deg2rad,
    rad2deg,
    p_SHG_fun,
    s_SHG_fun,
    p_TPF_fun,
    s_TPF_fun,
    E,
)
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np

column_names_replace = dict(
    zip(
        ["Plate.row", "Plate.col", "Molecule.Name", "Concentration [uM]"],
        ["Row", "Column", "Source Substance", "Source Concentration"],
    )
)


def label_rows(df):
    choose_cols = ["Source Row", "Source Column"]
    well_labels = [
        "{:s}{:d}".format(*[row[h] for h in choose_cols])
        for index, row in df[choose_cols].iterrows()
    ]
    return df.set_index(pd.Series(well_labels))


def df_list_to_array(df, repeat_x=True):
    """ Converts a MultiIndex DataFrame (pivot table) to arrays

    The data entries in the input pivot table is meant to be of the type
    'list', so they are not aggregated / reduced. This is meant to be used
    for getting individual entries that are replicates of the same grouped
    category (e.g. original data set was reduced by passing 'list' to the
    `aggfunc` argument of `df.pivot_table`)

    The input DataFrame must have the following arrangements:

                                            ____________COLUMNS___________
    ________________________________________| ch#1 |  ch#2 | ch#3 | ch#4 |
    Source Substance | Source Concentration |      |       |      |      |
    -----------------|----------------------|----------------------------|
       dmso          |      0.00            | [..] | [..]  | [..] | [..] |
                     |                      | [..] | [..]  | [..] | [..] |
                     |                      | [..] | [..]  | [..] | [..] |
                     |                      | [..] | [..]  | [..] | [..] |
      cmpd1          |      0.13            | [..] | [..]  | [..] | [..] |
                     |      0.26            | [..] | [..]  | [..] | [..] |
                     |      0.52            | [..] | [..]  | [..] | [..] |
                     |      1.04            | [..] | [..]  | [..] | [..] |

    The returned output is a dictionary with keys derived from items in the
    first column, (e.g. "Source Substance"), followed by an array of
    "Source Concentration"

    """

    # number of rows/index
    Nindex = len(df.index.levels)

    if Nindex != 2:
        raise ValueError("There must be 2 exactly 2 row levels.")

    ret_dict = {}

    index_names = df.index.names

    # go through each 'Compound' at level 0
    for level_name in df.index.get_level_values(0).unique():
        wrk_dict = {}
        for channel_name in df.columns.tolist():
            # get one column, 'Channel', from the sliced DataFrame
            subdf = df.xs(level_name, level=0, axis=0)[channel_name]
            # the remaining level
            x = subdf.index.to_numpy()
            y = np.array([x for x in subdf])
            # column of the array is the # of replicates
            if repeat_x:
                x = np.repeat(x[:, np.newaxis], y.shape[1], axis=1)
            # assign to dictionary
            wrk_dict[index_names[1]] = x
            wrk_dict[channel_name] = y
        # back to 'Compound' loop
        ret_dict[level_name] = wrk_dict

    return ret_dict


def fit_data_set_v2(df, output_pdf):
    # input dataframe must not have test compounds, otherwise
    # the CRC fitting would fail because the concentrations are constant
    data_headers = [
        "%ΔP-SHG",
        "%ΔS-SHG",
        "P-FLcorr",
        "S-FLcorr",
        "SHGratio",
        "TPFratio",
        "Angle",
        "distribution",
    ]

    header_locs = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
    ]

    ax_coords = list(zip(data_headers, header_locs))

    txtboxfmt = r"$K_D = {:6.2f} \pm {:6.2f} \mu M, ({:5.1f}\%)$"
    boxprop = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.25}
    rep_color = ("#286ede", "#0fa381")

    with PdfPages(output_pdf) as mpdf:

        for compound, subdf in df.groupby(["Source Substance"]):
            print("\rWorking on compound ... {:s}".format(compound), end="")
            source_plates = subdf["Source Plate ID"].unique().tolist()
            # set1
            subset1 = subdf[subdf["Source Plate ID"] == source_plates[0]].copy()
            subset2 = subdf[subdf["Source Plate ID"] == source_plates[1]].copy()
            subset1.sort_values("Source Concentration", inplace=True)
            subset2.sort_values("Source Concentration", inplace=True)
            fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(8.5, 11))
            fig.suptitle(compound)

            for channel, (i, j) in ax_coords:
                x1 = subset1["Source Concentration"].values
                x2 = subset2["Source Concentration"].values
                y1 = subset1[channel].values
                y2 = subset2[channel].values

                legend_item1, = ax[i, j].plot(
                    x1,
                    y1,
                    ".",
                    c=rep_color[0],
                    label="{:s}".format(source_plates[0]),
                )
                legend_item2, = ax[i, j].plot(
                    x1,
                    y2,
                    ".",
                    c=rep_color[1],
                    label="{:s}".format(source_plates[1]),
                )

                ax[i, j].plot(x1, (y1 + y2) / 2.0, "o", c="#fa880f")

                # now do fitting
                x_rep = np.array([x1, x2])
                y_rep = np.array([y1, y2])

                if channel in ["%ΔP-SHG", "%ΔS-SHG"]:
                    # for SHG signals
                    binding_model = BindingCurve.Hyperbolic(
                        x_rep.ravel(), y_rep.ravel(), modality="SHG"
                    )
                    try:
                        binding_model.fit()
                    except ValueError:
                        binding_model = None
                else:
                    # for anything else
                    binding_model = BindingCurve.Hyperbolic(
                        x_rep.ravel(), y_rep.ravel(), modality="TPF"
                    )
                    try:
                        binding_model.fit()
                    except ValueError:
                        binding_model = None

                # fitting done, overlay the result
                if binding_model is not None and binding_model.optres.success:
                    xsmooth = logspace(x1.min(), x1.max())
                    ax[i, j].plot(
                        xsmooth, binding_model.model(xsmooth), "k-", lw=2
                    )

                    Kd_value = binding_model.optres.params["Kd"].value
                    Kd_stderr = binding_model.optres.params["Kd"].stderr

                    if Kd_value is None or Kd_stderr is None:
                        txtstr = None
                    else:
                        Kd_error = 100.0 * (Kd_stderr / Kd_value)
                        txtstr = txtboxfmt.format(Kd_value, Kd_stderr, Kd_error)
                else:
                    txtstr = None

                if txtstr is not None:
                    ax[i, j].text(
                        0.05,
                        0.8,
                        txtstr,
                        transform=ax[i, j].transAxes,
                        fontsize=9,
                        bbox=boxprop,
                    )

                ax[i, j].set_xscale("log")
                ax[i, j].set_ylabel("{:s}".format(channel))

            fig.legend(
                [legend_item1, legend_item2],
                labels=[legend_item1.get_label(), legend_item2.get_label()],
                fancybox=True,
                shadow=True,
                bbox_to_anchor=(0.7, 0.88, 0.15, 0.1),
            )

            mpdf.savefig()
            plt.close()


def fit_data_set(data, output_pdf, key_pair):
    """ Curve fitting pipeline for June 2019 hit follow-up """
    compound_list = list(data.keys())

    # ctrl_compounds = [c for c in compound_list if c.startswith('ctrl_')]
    test_compounds = [c for c in compound_list if not c.startswith("ctrl_")]

    duplicates = key_pair
    data_headers = [
        "%ΔP-SHG",
        "%ΔS-SHG",
        "P-FLcorr",
        "S-FLcorr",
        "SHGratio",
        "TPFratio",
        "Angle",
        "distribution",
    ]
    header_locs = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
    ]
    ax_coords = list(zip(data_headers, header_locs))

    # data is now grouped for fitting
    txtboxfmt = r"$K_D = {:.2f} \pm {:.2f} \mu M, ({:.1f}\%)$"
    boxprop = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.25}
    rep_color = ("#286ede", "#0fa381")
    compound_Kd_dict = {}

    with PdfPages(output_pdf) as mpdf:
        for compound in test_compounds:
            print("Processing ... {:s}".format(compound))
            Kd_dict = {}
            fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(8.5, 11))
            fig.suptitle(compound)
            sub_data = data[compound]
            for channel, (i, j) in ax_coords:
                x_combined = []
                y_combined = []
                legend_items = []

                for plate_ID in duplicates:
                    x_data = sub_data[plate_ID]["Source Concentration"]
                    y_data = sub_data[plate_ID][channel]
                    x_combined.append(x_data)
                    y_combined.append(y_data)
                    legend_item, = ax[i, j].plot(
                        x_data,
                        y_data,
                        ".",
                        c=rep_color[plate_ID == duplicates[0]],
                        label="{:s}".format(plate_ID),
                    )
                    legend_items.append(legend_item)

                # do fitting
                x_arr = np.array(x_combined)
                y_arr = np.array(y_combined)
                x_mean = x_combined[0]
                y_mean = (y_combined[0] + y_combined[1]) / 2.0
                ax[i, j].plot(x_mean, y_mean, "o", c="#fa880f")

                if channel in ["%ΔP-SHG", "%ΔS-SHG"]:
                    binding_model = BindingCurve.Hyperbolic(
                        x_arr.ravel(), y_arr.ravel(), modality="SHG"
                    )
                    try:
                        binding_model.fit()
                    except ValueError:
                        binding_model = None
                else:

                    binding_model = BindingCurve.Hyperbolic(
                        x_arr.ravel(), y_arr.ravel(), modality="TPF"
                    )
                    try:
                        binding_model.fit()
                    except ValueError:
                        binding_model = None

                if binding_model is not None and binding_model.optres.success:
                    # do filtering here for fitting quality
                    xsmooth = logspace(x_arr.min(), x_arr.max())
                    ax[i, j].plot(
                        xsmooth, binding_model.model(xsmooth), "k-", lw=2
                    )
                    Kd_value = binding_model.optres.params["Kd"].value
                    Kd_stderr = binding_model.optres.params["Kd"].stderr
                    Kd_error = 100.0 * (Kd_stderr / Kd_value)
                    Kd_CV = Kd_stderr / Kd_value
                    txtstr = txtboxfmt.format(Kd_value, Kd_stderr, Kd_error)
                    Kd_dict[channel] = [Kd_value, Kd_stderr]
                else:
                    txtstr = None
                    Kd_dict[channel] = [np.nan, np.nan]

                if txtstr is not None:
                    ax[i, j].text(
                        0.05,
                        0.8,
                        txtstr,
                        transform=ax[i, j].transAxes,
                        fontsize=9,
                        bbox=boxprop,
                    )
                ax[i, j].set_xlim([1e-2, 25.0])
                ax[i, j].set_xscale("log")
                ax[i, j].set_ylabel("{:s}".format(channel))

            fig.legend(
                legend_items[0:2],
                labels=[i.get_label() for i in legend_items[0:2]],
                fancybox=True,
                shadow=True,
                bbox_to_anchor=(0.7, 0.9, 0.15, 0.1),
            )

            mpdf.savefig()
            plt.close()

            compound_Kd_dict[compound] = Kd_dict

    return compound_Kd_dict


def eval_processed(
    df,
    lb_fun,
    ub_fun,
    pSHG_unlabeled=6550.0,
    sSHG_unlabeled=770.0,
    inplace=True,
):
    """ evaluates processed DataFrames for failure modes

    Two failure modes are considered: SHG background correction and angular
    solution.

    Args:
        lb_fun(callable): callable with signature lb_fun(x), where x is the
            TPF ratio. This should return the lower-bound for SHG ratio.
        ub_fun(callable): callable with signature ub_fun(x), where x is the
            TPF ratio. This should return the upper-bound for SHG ratio.
        pSHG_unlabeled(float): this is the value for unlabeled protein attached
            to the surface in the P-SHG channel.
        sSHG_unlabeled(float): this is the value for unlabeled protein attached
            to the surface in the S-SHG channel.

    """
    shg_bg_fails = []
    solvable = []
    series_id = []

    for i, item in df[["P-SHG", "S-SHG", "SHGratio", "TPFratio"]].iterrows():
        if item["P-SHG"] <= pSHG_unlabeled or item["S-SHG"] <= sSHG_unlabeled:
            shg_bg_fails.append(True)
            # if background can't be corrected, we also can't get angles
            # because we have no SHG ratio
            solvable.append(False)
        else:
            # passed background subtraction check, now check for solutions
            shg_bg_fails.append(False)

            x, y = item["TPFratio"], item["SHGratio"]
            if lb_fun(x) < y < ub_fun(x):
                solvable.append(True)
            else:
                solvable.append(False)

        series_id.append(i)

    if inplace:
        df["SHG_BG_fail"] = pd.Series(shg_bg_fails, index=series_id)
        df["Angular_solution"] = pd.Series(solvable, index=series_id)
        return None
    else:
        copied_df = df.copy()
        copied_df["SHG_BG_fail"] = pd.Series(shg_bg_fails, index=series_id)
        copied_df["Angular_solution"] = pd.Series(solvable, index=series_id)
    return copied_df


def simulate_data(angle, distribution):
    """ compute 4-channel quantities for a given angle and distribution

    Args:
        angle(float): mean tilt angle of the probe.
        distribution(float): distribution width or standard deviation.

    Returns:
        <cos**3(θ)>**2, <sin**2(θ)·cos(θ)>**2, <sin**4(θ)·cos**2(θ)>, <sin**6(θ)>
    """
    θ = np.linspace(0.0, np.pi, num=500)
    I_ppp = E(θ, deg2rad(angle), deg2rad(distribution), p_SHG_fun) ** 2
    I_pss = E(θ, deg2rad(angle), deg2rad(distribution), s_SHG_fun) ** 2
    F_pp = E(θ, deg2rad(angle), deg2rad(distribution), p_TPF_fun)
    F_ss = E(θ, deg2rad(angle), deg2rad(distribution), s_TPF_fun)
    return I_ppp, I_pss, F_pp, F_ss


def screen_qc(grp, neg_ctrl_pos, pos_ctrl_pos):
    negative_ctrl = grp.loc[
        grp["Well coordinates"].isin(neg_ctrl_pos), "%SHG change"
    ]
    positive_ctrl = grp.loc[
        grp["Well coordinates"].isin(pos_ctrl_pos), "%SHG change"
    ]

    zprime = 1 - (
        3
        * (positive_ctrl.std() + negative_ctrl.std())
        / (positive_ctrl.mean() - negative_ctrl.mean())
    )

    zrobust = 1 - (
        3
        * (positive_ctrl.mad() - negative_ctrl.mad())
        / (positive_ctrl.median() - negative_ctrl.median())
    )

    retdict = {
        "Z-prime": zprime,
        "robust Z-prime": zrobust,
    }

    return pd.Series(retdict)


def compound_summarizer(grp):
    """ Binding-curve based aggregator and calculate all quality metrics

    This function is mean to be used as part of a chained operation from
    a dataframe. Otherwise, it takes a dataframe with all of the 'primary' and
    'secondary' quantities (observation, corrected observation, or derived
    quantities like angles and distribution)

    Example use::

        data = pd.read_excel('./Analysis/All_compounds_summary.xlsx')
        # summarize non control datasets

        # dont analyze control compounds
        test_set = [cmpd for cmpd in data['Source Substance'].unique().tolist()
                   if not cmpd.startswith('ctrl_')]
        test_df = data[data['Source Substance'].isin(test_set)]

        # here test_df is a DataFrame containing columns.
        # The sub-indices are retained in the output, keeping the 'Source Plate
        # ID' separate in the output.
        summarized_df = (
            test_df.set_index(['Source Substance'])
            .groupby(by=['Source Substance'], axis=0)
            .apply(compound_summarizer)
        )

    """

    x = grp["Source Concentration"]
    y1 = grp["%ΔP-SHG"]
    y2 = grp["%ΔS-SHG"]
    y3 = grp["P-FLcorr"]
    y4 = grp["S-FLcorr"]

    N_proc_fail = grp["SHG_BG_fail"].sum()
    N_solved = grp["Angular_solution"].sum()
    Ndata = grp["SHG_BG_fail"].count()

    # data fitting bit
    m1 = BindingCurve.Hyperbolic(x.values, y1.values, modality="SHG")
    m1.fit()

    m2 = BindingCurve.Hyperbolic(x.values, y2.values, modality="SHG")
    m2.fit()

    m3 = BindingCurve.Hyperbolic(x.values, y3.values, modality="TPF")
    m3.fit()

    m4 = BindingCurve.Hyperbolic(x.values, y4.values, modality="TPF")
    m4.fit()

    def _CV_score(x, y):
        cv = x / y if x is not None else 1e8
        return cv

    Kd1, Kd1_CV = (
        m1.optres.params["Kd"].value,
        _CV_score(m1.optres.params["Kd"].stderr, m1.optres.params["Kd"].value),
    )

    Kd2, Kd2_CV = (
        m2.optres.params["Kd"].value,
        _CV_score(m2.optres.params["Kd"].stderr, m2.optres.params["Kd"].value),
    )

    Kd3, Kd3_CV = (
        m3.optres.params["Kd"].value,
        _CV_score(m3.optres.params["Kd"].stderr, m3.optres.params["Kd"].value),
    )

    Kd4, Kd4_CV = (
        m4.optres.params["Kd"].value,
        _CV_score(m4.optres.params["Kd"].stderr, m4.optres.params["Kd"].value),
    )

    # Physicality score bit
    vec_angles = grp["Angle"]
    vec_dists = grp["distribution"]

    # also fit a curve to the angle and distribution
    m5 = BindingCurve.Hyperbolic(x.values, vec_angles.values, modality="TPF")
    m5.fit()
    m6 = BindingCurve.Hyperbolic(x.values, vec_dists.values, modality="TPF")
    m6.fit()

    Kd5, Kd5_CV = (
        m5.optres.params["Kd"].value,
        _CV_score(m5.optres.params["Kd"].stderr, m5.optres.params["Kd"].value),
    )

    Kd6, Kd6_CV = (
        m6.optres.params["Kd"].value,
        _CV_score(m6.optres.params["Kd"].stderr, m6.optres.params["Kd"].value),
    )

    calc_Ippp = np.zeros(Ndata)
    calc_Ipss = np.zeros(Ndata)
    calc_Fpp = np.zeros(Ndata)
    calc_Fss = np.zeros(Ndata)

    # convert angles to
    # compute intensities from angles and distribution
    # convert to numpy to facilitate numerical indexing
    vec_angles = vec_angles.values
    vec_dists = vec_dists.values

    for n in range(Ndata):
        if ~np.isnan(vec_angles[n]) and ~np.isnan(vec_dists[n]):
            calc_Ippp[n], calc_Ipss[n], calc_Fpp[n], calc_Fss[
                n
            ] = simulate_data(vec_angles[n], vec_dists[n])
        else:
            calc_Ippp[n] = calc_Ipss[n] = calc_Fpp[n] = calc_Fss[n] = np.nan

    mask1 = ~np.isnan(calc_Ippp) & ~np.isnan(grp["P-SHGcorr"])
    mask2 = ~np.isnan(calc_Ipss) & ~np.isnan(grp["S-SHGcorr"])
    mask3 = ~np.isnan(calc_Fpp) & ~np.isnan(grp["P-FLcorr"])
    mask4 = ~np.isnan(calc_Fss) & ~np.isnan(grp["S-FLcorr"])

    m1, b1, rval1, pval1, m_err1 = linregress(
        calc_Ippp[mask1], grp["P-SHGcorr"][mask1]
    )
    m2, b2, rval2, pval2, m_err2 = linregress(
        calc_Ipss[mask1], grp["S-SHGcorr"][mask2]
    )
    m3, b3, rval3, pval3, m_err3 = linregress(
        calc_Fpp[mask1], grp["P-FLcorr"][mask3]
    )
    m4, b4, rval4, pval4, m_err4 = linregress(
        calc_Fss[mask1], grp["S-FLcorr"][mask4]
    )

    logprob_physical = (
        (1.0 - rval1) ** 2
        + (1.0 - rval2) ** 2
        + (1.0 - rval3) ** 2
        + (1.0 - rval4) ** 2
    )

    # all rvalues independent prod(prob(rvals)), most stringent
    prob_physical_v1 = np.exp(-logprob_physical)
    # prob(rval), either one (most relaxed). Uniform averaging for all
    # probabilities
    prob_physical_v2 = (
        np.exp(-(1 - rval1) ** 2)
        + np.exp(-(1 - rval2) ** 2)
        + np.exp(-(1 - rval3) ** 2)
        + np.exp(-(1 - rval4) ** 2)
    ) / 4.0

    # prob(SHG) and prob(FL), middle
    prob_physical_v3 = (
        (np.exp(-(1 - rval1) ** 2) + np.exp(-(1 - rval2) ** 2))
        / 2.0
        * (np.exp(-(1 - rval3) ** 2) + np.exp(-(1 - rval4) ** 2))
        / 2.0
    )

    retdict = {
        "Kd (P-SHG)": Kd1,
        "Kd (S-SHG)": Kd2,
        "Kd (P-FLcorr)": Kd3,
        "Kd (S-FLcorr)": Kd4,
        "Kd (Angles)": Kd5,
        "Kd (distribution)": Kd6,
        "CV (P-SHG)": Kd1_CV,
        "CV (S-SHG)": Kd2_CV,
        "CV (P-FLcorr)": Kd3_CV,
        "CV (S-FLcorr)": Kd4_CV,
        "CV (Angles)": Kd5_CV,
        "CV (distribution)": Kd6_CV,
        "R (P-SHG)": rval1,
        "R (S-SHG)": rval2,
        "R (P-FL)": rval3,
        "R (S-FL)": rval4,
        "Prob(physical)_v1": prob_physical_v1,
        "Prob(physical)_v2": prob_physical_v2,
        "Prob(physical)_v3": prob_physical_v3,
        "#BG fail": N_proc_fail,
        "#Solved": N_solved,
        "Count": Ndata,
    }

    return pd.Series(retdict, dtype=object)


def eval_scores(grp):
    """ an aggregator function to incorporate replicate error into quality scores

    Usage ::

        # read all of the compile dataset
        data = pd.read_excel("./Processed/All_experiments.xlsx", index_col=0)

        # index data by compound and plate ID
        indexed_data = data\
        .set_index(['Source Substance', 'Source Plate ID'])
        .sort_values(by=["Source Substance","Source Plate ID"])

        data_scores = indexed_data\
        .groupby(['Source Substance'])\
        .apply(eval_scores).unstack()

    """
    current_compound = grp.index.get_level_values(0)[0]
    # get 'Source Plate ID'
    plate_id = grp.index.get_level_values("Source Plate ID").unique().tolist()

    # split dataset according to 'Source Plate ID'
    rep1 = grp.xs(plate_id[0], level="Source Plate ID", axis=0).set_index(
        "Source Concentration"
    )
    rep2 = grp.xs(plate_id[1], level="Source Plate ID", axis=0).set_index(
        "Source Concentration"
    )

    # dictionary of all channels
    ret_scores = {}

    # columns containing data to analyze
    data_columns = [
        "%ΔP-SHG",
        "%ΔS-SHG",
        "P-FLcorr",
        "S-FLcorr",
        "SHGratio",
        "TPFratio",
        "Angle",
        "distribution",
    ]

    angle_fails = grp["Angle"].isna().sum()
    total_counts = grp["Angle"].isna().count()
    percent_fail = angle_fails / total_counts

    for column in data_columns:
        # reproducibility score
        rep1vals = rep1[column]
        rep2vals = rep2.loc[rep1.index][column]

        mask = ~np.isnan(rep1vals) & ~np.isnan(rep2vals)
        m, b, rval, pval, merr = linregress(rep1vals[mask], rep2vals[mask])

        if np.isnan(rval):
            print(
                (
                    "Couldn't fit line between replicates "
                    "in column {:s} of {:s}!"
                ).format(column, current_compound)
            )
        # Fit a CRC to current column
        # this concatenates along row so 2xN
        x = np.array(
            [
                rep1.index.get_level_values("Source Concentration").to_numpy(),
                rep2.index.get_level_values("Source Concentration").to_numpy(),
            ]
        )
        y = np.array([rep1[column], rep2[column]])

        if column == "SHGratio":
            m_modality = "TPF"
        else:
            m_modality = "SHG" if "SHG" in column else "TPF"

        model = BindingCurve.Hyperbolic(
            x.ravel(), y.ravel(), modality=m_modality
        )

        model.fit()

        if model.optres.errorbars:
            Kd_val = model.optres.params["Kd"].value
            Kd_err = model.optres.params["Kd"].stderr
            Kd_CV = Kd_err / Kd_val
        else:
            Kd_val = 1e6
            Kd_CV = 1e6

        # calculate the dynamic range of the data
        data_range = y.max() - y.min()

        ret_scores[column] = pd.Series(
            {
                "dynamic_range": data_range,
                "slope": m,
                "r-value": rval,
                "Kd": Kd_val,
                "StdErr(Kd)": Kd_err,
                "CV(Kd)": Kd_CV,
                r"% solution fail": percent_fail,
            }
        )

    return pd.DataFrame.from_dict(ret_scores, orient="columns")


def plot_physical_scores(df, output_pdf):

    txtboxfmt = r"$R-value: {:.2f}, Error : {:.1f}%$"
    boxprop = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.25}

    with PdfPages(output_pdf) as mpdf:
        for compound, subdf in df.groupby(["Source Substance"]):
            print("\rWorking on compound ... {:s} ".format(compound), end="")

            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8.5, 11))
            fig.suptitle(compound)

            vec_angles = subdf["Angle"].to_numpy()
            vec_dists = subdf["distribution"].to_numpy()

            Ndata = vec_angles.size

            calc_Ippp = np.zeros(Ndata)
            calc_Ipss = np.zeros(Ndata)
            calc_Fpp = np.zeros(Ndata)
            calc_Fss = np.zeros(Ndata)

            for n in range(Ndata):
                if ~np.isnan(vec_angles[n]) and ~np.isnan(vec_dists[n]):
                    calc_Ippp[n], calc_Ipss[n], calc_Fpp[n], calc_Fss[
                        n
                    ] = simulate_data(vec_angles[n], vec_dists[n])
                else:
                    calc_Ippp[n] = calc_Ipss[n] = calc_Fpp[n] = calc_Fss[
                        n
                    ] = np.nan

            mask1 = ~np.isnan(calc_Ippp) & ~np.isnan(subdf["P-SHGcorr"])
            mask2 = ~np.isnan(calc_Ipss) & ~np.isnan(subdf["S-SHGcorr"])
            mask3 = ~np.isnan(calc_Fpp) & ~np.isnan(subdf["P-FLcorr"])
            mask4 = ~np.isnan(calc_Fss) & ~np.isnan(subdf["S-FLcorr"])

            m1, b1, rval1, pval1, m_err1 = linregress(
                calc_Ippp[mask1], subdf["P-SHGcorr"][mask1]
            )
            report1 = txtboxfmt.format(rval1, 100.0 * m_err1 / m1)

            m2, b2, rval2, pval2, m_err2 = linregress(
                calc_Ipss[mask2], subdf["S-SHGcorr"][mask2]
            )
            report2 = txtboxfmt.format(rval2, 100.0 * m_err2 / m2)

            m3, b3, rval3, pval3, m_err3 = linregress(
                calc_Fpp[mask3], subdf["P-FLcorr"][mask3]
            )
            report3 = txtboxfmt.format(rval3, 100.0 * m_err3 / m3)

            m4, b4, rval4, pval4, m_err4 = linregress(
                calc_Fss[mask4], subdf["S-FLcorr"][mask4]
            )
            report4 = txtboxfmt.format(rval4, 100.0 * m_err4 / m4)

            # plot the xy-data
            ax[0, 0].plot(calc_Ippp, subdf["P-SHGcorr"], ".")
            ax[0, 1].plot(calc_Ipss, subdf["S-SHGcorr"], ".")
            ax[1, 0].plot(calc_Fpp, subdf["P-FLcorr"], ".")
            ax[1, 1].plot(calc_Fss, subdf["S-FLcorr"], ".")
            # plot the linear regression
            ax[0, 0].plot(calc_Ippp, calc_Ippp * m1 + b1, "k-")
            ax[0, 1].plot(calc_Ipss, calc_Ipss * m2 + b2, "k-")
            ax[1, 0].plot(calc_Fpp, calc_Fpp * m3 + b3, "k-")
            ax[1, 1].plot(calc_Fss, calc_Fss * m4 + b4, "k-")

            # plot annotations
            ax[0, 0].text(
                0.05, 0.08, report1, bbox=boxprop, transform=ax[0, 0].transAxes
            )
            ax[0, 0].set_xlabel(r"calculated $I_{ppp}$")
            ax[0, 0].set_ylabel(r"observed $I_{ppp}$")

            ax[0, 1].text(
                0.05, 0.08, report2, bbox=boxprop, transform=ax[0, 1].transAxes
            )

            ax[0, 1].set_xlabel(r"calculated $I_{pss}$")
            ax[0, 1].set_ylabel(r"observed $I_{pss}$")

            ax[1, 0].text(
                0.05, 0.08, report3, bbox=boxprop, transform=ax[1, 0].transAxes
            )

            ax[1, 0].set_xlabel(r"calculated $F_{pp}$")
            ax[1, 0].set_ylabel(r"observed $F_{pp}$")

            ax[1, 1].text(
                0.05, 0.08, report4, bbox=boxprop, transform=ax[1, 1].transAxes
            )

            ax[1, 1].set_xlabel(r"calculated $F_{ss}$")
            ax[1, 1].set_ylabel(r"observed $F_{ss}$")

            mpdf.savefig()
            plt.close()
