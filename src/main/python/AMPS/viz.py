import numpy as np
import matplotlib.pyplot as plt
from AMPS import AMPSexperiment
from pandas import DataFrame
from . import _shglut, _tpflut


def place_in_range(vec, frac):
    vecmax = vec.max()
    vecmin = vec.min()
    return vecmin + (vecmax - vecmin) * frac


def strip_chart(data, grp, ax, column_name="TPFratio"):

    fracs, idx = np.unique(data["frac_labeled"], return_inverse=True)
    jittered_idx = idx + (np.random.random(idx.size) - 0.4) * 0.2
    xbar = np.arange(fracs.size)

    fraclabels = [f"{v*100:.1f}%" for v in fracs]

    ax.bar(
        xbar,
        grp[column_name]["mean"],
        yerr=grp[column_name]["std"],
        linewidth=0.8,
        edgecolor="black",
        capsize=2.5,
        width=0.8,
        color="pink",
        alpha=0.5,
        error_kw={"elinewidth": 0.75},
    )
    ax.plot(jittered_idx, data[column_name], "r.", alpha=0.5, ms=8)

    ax.set_xticks(xbar)
    ax.set_xticklabels(fraclabels)
    ax.set_xlabel("percent labeled")
    ax.set_ylabel(f"{column_name}")


def linearity_check(grp, ax, twinax=None):

    if twinax is None:
        ax_twin = ax.twinx()
    else:
        ax_twin = twinax

    x = grp.index.values
    yP = grp["P-FLcorr"]["mean"].values
    yS = grp["S-FLcorr"]["mean"].values
    sy_P = grp["P-FLcorr"]["std"].values
    sy_S = grp["S-FLcorr"]["std"].values
    m1, _, _, _ = np.linalg.lstsq(x[:, np.newaxis], yP, rcond=-1)
    m2, _, _, _ = np.linalg.lstsq(x[:, np.newaxis], yS, rcond=-1)
    ax.errorbar(
        x,
        yP,
        yerr=sy_P,
        fmt="o",
        mew=0.5,
        mfc="teal",
        mec="k",
        alpha=0.5,
        ecolor="k",
        elinewidth=0.75,
        capsize=2,
    )
    ax_twin.errorbar(
        x,
        yS,
        yerr=sy_S,
        fmt="o",
        mew=0.5,
        mfc="blue",
        mec="k",
        alpha=0.5,
        ecolor="k",
        elinewidth=0.75,
        capsize=2,
    )

    # plot linear fits
    ax.plot(x, m1 * x, "-", lw=2, color="teal")
    ax_twin.plot(x, m2 * x, "-", lw=2, color="blue")

    ax.set_ylabel("P-FLcorr", color="teal")
    ax.tick_params(axis="y", labelcolor="teal")
    ax_twin.set_ylabel("S-FLcorr", color="blue")
    ax_twin.tick_params(axis="y", labelcolor="blue")

    ax.grid(False)
    ax_twin.grid(False)

    ax.set_xlabel("fraction labeled")
    ax.set_title("fluorescence linearity")


def square_check(grp, ax):
    # copy the existing axis
    ax_twin = ax.twinx()

    x = grp.index.values
    yP = grp["P-SHGcorr"]["mean"].values
    yS = grp["S-SHGcorr"]["mean"].values
    sy_P = grp["P-SHGcorr"]["std"].values
    sy_S = grp["S-SHGcorr"]["std"].values

    ax.errorbar(
        x,
        yP,
        yerr=sy_P,
        fmt="o",
        mew=0.5,
        mfc="teal",
        mec="k",
        alpha=0.5,
        ecolor="k",
        elinewidth=0.75,
        capsize=2,
    )

    ax_twin.errorbar(
        x,
        yS,
        yerr=sy_S,
        fmt="o",
        mew=0.5,
        mfc="blue",
        mec="k",
        alpha=0.5,
        ecolor="k",
        elinewidth=0.75,
        capsize=2,
    )

    ax.set_ylabel("P-SHGcorr", color="teal")
    ax.tick_params(axis="y", labelcolor="teal")
    ax_twin.set_ylabel("S-SHGcorr", color="blue")
    ax_twin.tick_params(axis="y", labelcolor="blue")

    ax.grid(False)
    ax_twin.grid(False)

    ax.set_xlabel("fraction labeled")
    ax.set_title("concentration vs SHG")


def overview(
    data, figsize=(8, 6.25), experiment=None, fighandles=None, twin_ax=None
):
    """ summarize overview of 4-channel data

    This function may change in the future

    Args:
        data(DataFrame): raw 4-channel data input

    """
    stats = data.groupby("frac_labeled").agg(["mean", "std"])

    if fighandles is None:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        returnfig = True
    else:
        fig, ax = fighandles
        returnfig = False

    ax[0, 0].plot(
        data["P-FLcorr"].values.ravel(),
        data["P-SHG"].values.ravel(),
        "r.",
        alpha=0.3,
    )
    ax[0, 1].plot(
        data["S-FLcorr"].values.ravel(),
        data["S-SHG"].values.ravel(),
        "r.",
        alpha=0.3,
    )

    ax[0, 0].errorbar(
        stats["P-FLcorr"]["mean"],
        stats["P-SHG"]["mean"],
        xerr=stats["P-FLcorr"]["std"],
        yerr=stats["P-SHG"]["std"],
        fmt="o",
        mfc="red",
        mec="black",
        mew=0.5,
        ecolor="red",
        elinewidth=0.75,
        capsize=2.0,
    )

    ax[0, 1].errorbar(
        stats["S-FLcorr"]["mean"],
        stats["S-SHG"]["mean"],
        xerr=stats["S-FLcorr"]["std"],
        yerr=stats["S-SHG"]["std"],
        fmt="o",
        mfc="red",
        mec="black",
        mew=0.5,
        ecolor="red",
        elinewidth=0.75,
        capsize=2.0,
    )

    if experiment is not None:
        # overlay the fit result
        if experiment.Pphases.optres.success:
            ax[0, 0].plot(
                experiment.Pphases.fit.x,
                experiment.Pphases.fit.y,
                "-",
                c="darkred",
                lw=1.5,
                zorder=-100,
            )
            _deg = np.rad2deg(experiment.Pphases.optres.params["delphi"].value)

            if experiment.Pphases.optres.params["delphi"].stderr is not None:
                _degerr = np.rad2deg(
                    experiment.Pphases.optres.params["delphi"].stderr
                )
            else:
                _degerr = np.nan

            ax[0, 0].text(
                0.05,
                0.85,
                f"{_deg:.0f} ± {_degerr:.0f} ˚",
                fontsize=12,
                transform=ax[0, 0].transAxes
            )

        if experiment.Sphases.optres.success:
            ax[0, 1].plot(
                experiment.Sphases.fit.x,
                experiment.Sphases.fit.y,
                "-",
                c="darkred",
                lw=1.5,
                zorder=-100,
            )
            _deg = np.rad2deg(experiment.Sphases.optres.params["delphi"].value)

            if experiment.Sphases.optres.params["delphi"].stderr is not None:
                _degerr = np.rad2deg(
                    experiment.Sphases.optres.params["delphi"].stderr
                )
            else:
                _degerr = np.nan

            reptext = f"{_deg:.0f} ± {_degerr:.0f} ˚"
            ax[0, 1].text(
                0.05,
                0.85,
                f"{_deg:.0f} ± {_degerr:.0f} ˚",
                fontsize=12,
                transform=ax[0, 1].transAxes,
            )

    ax[0, 0].set_xlabel("P-FLcorr")
    ax[0, 0].set_ylabel("P-SHG")

    ax[0, 1].set_xlabel("S-FLcorr")
    ax[0, 1].set_ylabel("S-SHG")

    strip_chart(data, stats, ax[1, 0])
    ax[1, 0].tick_params(axis="x", which="major", labelsize=7)

    if twin_ax is None:
        twin_ax = ax[1, 1].twinx()

    linearity_check(stats, ax[1, 1], twinax=twin_ax)

    fig.tight_layout()

    if returnfig:
        return fig, ax


def overview2(experiment, figsize=(8, 9.4)):
    """ summarize overview of 4-channel data

    This function will plot an analyzed AMPS experiment object.
    It assumes all of the quantities are already present in data.

    Args:
        data(DataFrame): raw 4-channel data input

    """
    stats = experiment.grpdf
    data = experiment.data

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=figsize)

    ax[0, 0].plot(
        data["P-FLcorr"].values.ravel(),
        data["P-SHG"].values.ravel(),
        "r.",
        alpha=0.3,
    )
    ax[0, 1].plot(
        data["S-FLcorr"].values.ravel(),
        data["S-SHG"].values.ravel(),
        "r.",
        alpha=0.3,
    )

    ax[0, 0].errorbar(
        stats["P-FLcorr"]["mean"],
        stats["P-SHG"]["mean"],
        xerr=stats["P-FLcorr"]["std"],
        yerr=stats["P-SHG"]["std"],
        fmt="o",
        mfc="red",
        mec="black",
        mew=0.5,
        ecolor="red",
        elinewidth=0.75,
        capsize=2.0,
    )

    ax[0, 1].errorbar(
        stats["S-FLcorr"]["mean"],
        stats["S-SHG"]["mean"],
        xerr=stats["S-FLcorr"]["std"],
        yerr=stats["S-SHG"]["std"],
        fmt="o",
        mfc="red",
        mec="black",
        mew=0.5,
        ecolor="red",
        elinewidth=0.75,
        capsize=2.0,
    )

    if experiment is not None:

        if experiment.Pphases is None:
            pass
        # overlay the fit result
        elif experiment.Pphases.optres.success:
            ax[0, 0].plot(
                experiment.Pphases.fit.x,
                experiment.Pphases.fit.y,
                "-",
                c="darkred",
                lw=1.5,
                zorder=-100,
            )

            # figure out position for text
            xpos = place_in_range(experiment.Pphases.fit.x, 0.1)
            ypos = place_in_range(experiment.Pphases.fit.y, 0.8)

            # convert to degrees
            _deg = np.rad2deg(experiment.Pphases.optres.params["delphi"].value)

            if experiment.Pphases.optres.params["delphi"].stderr is not None:
                _degerr = np.rad2deg(
                    experiment.Pphases.optres.params["delphi"].stderr
                )
            else:
                _degerr = np.nan

            ax[0, 0].text(
                xpos, ypos, f"{_deg:.0f} ± {_degerr:.0f} ˚", fontsize=12,
            )

        if experiment.Sphases is None:
            pass
        elif experiment.Sphases.optres.success:
            ax[0, 1].plot(
                experiment.Sphases.fit.x,
                experiment.Sphases.fit.y,
                "-",
                c="darkred",
                lw=1.5,
                zorder=-100,
            )
            # figure out position for text
            xpos = place_in_range(experiment.Sphases.fit.x, 0.1)
            ypos = place_in_range(experiment.Sphases.fit.y, 0.8)

            # convert to degrees
            _deg = np.rad2deg(experiment.Sphases.optres.params["delphi"].value)
            _degerr = np.rad2deg(
                experiment.Sphases.optres.params["delphi"].stderr
            )

            ax[0, 1].text(
                xpos, ypos, f"{_deg:.0f} ± {_degerr:.0f} ˚", fontsize=12,
            )

    ax[0, 0].set_xlabel("P-FLcorr")
    ax[0, 0].set_ylabel("P-SHG")

    ax[0, 1].set_xlabel("S-FLcorr")
    ax[0, 1].set_ylabel("S-SHG")

    strip_chart(data, stats, ax[1, 0])
    ax[1, 0].tick_params(axis="x", which="major", labelsize=11)

    linearity_check(stats, ax[1, 1])

    strip_chart(data, stats, ax[2, 0], column_name="SHGratio")
    ax[2, 0].tick_params(axis="x", which="major", labelsize=11)

    if ("angle" in data.columns) and ("distribution" in data.columns):

        labeled = data.query("frac_labeled == 1.0")
        rest = data.query("frac_labeled != 1.0")

        ax[2, 1].plot(
            rest["distribution"].values, rest["angle"].values, "o", c="gray"
        )
        ax[2, 1].plot(
            labeled["distribution"].values,
            labeled["angle"].values,
            "o",
            c="darkred",
        )

        ax[2, 1].set_xlabel("distribution")
        ax[2, 1].set_ylabel("angle")
        ax[2, 1].set_xlim([2.0, 70.0])
        ax[2, 1].set_ylim([0.0, 90.0])

    else:
        ax[2, 1].axis("off")

    fig.tight_layout()
    return fig, ax


def compare_column(
    experiment_dict,
    column_name,
    ax=None,
    frac_labeled=1.0,
    barwidth=0.8,
    barcolor="pink",
    markercolor="red",
    xtick_kwargs={},
):
    """ Generate bar plot for each column for each experiment """
    xlabels = []
    heights = []
    heights_err = []
    rawdata = []
    idxlist = []

    for i, (name, expt) in enumerate(experiment_dict.items()):

        if isinstance(expt, AMPSexperiment):
            data = expt.data
        elif isinstance(expt, DataFrame):
            data = expt

        xlabel = name
        grp = data.groupby("frac_labeled").agg(["mean", "std"])
        xlabels.append(xlabel)
        heights.append(grp.loc[frac_labeled][column_name]["mean"])
        heights_err.append(grp.loc[frac_labeled][column_name]["std"])
        _data = data[data["frac_labeled"] == frac_labeled][column_name].values
        # build raw data points in a list
        rawdata.append(_data)
        idxlist.append(np.repeat(i, _data.size))

    rawarr = np.concatenate(rawdata)
    idxarr = np.concatenate(idxlist)

    if ax is None:
        returnfig = True
        fig, ax = plt.subplots()
    else:
        returnfig = False

    xpos = np.arange(len(xlabels))
    jittered_idx = (
        idxarr + (np.random.random(idxarr.size) - 0.5) * barwidth / 2.5
    )
    ax.bar(
        xpos,
        heights,
        yerr=heights_err,
        width=barwidth,
        linewidth=0.8,
        capsize=2.5,
        edgecolor="black",
        color=barcolor,
        alpha=0.7,
        error_kw={"elinewidth": 0.8},
    )

    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels, **xtick_kwargs)

    ax.plot(jittered_idx, rawarr, "o", c=markercolor, alpha=0.5, ms=8)
    ax.set_title(f"{column_name}")
    ax.set_ylabel(f"{column_name}")

    if returnfig:
        return fig, ax


def compare_pair(experiment_dict, column1, column2, ax=None, frac_labeled=1.0):
    """ scatter plot between columns of experiments """

    if ax is None:
        fig, ax = plt.subplots()
        return_objs = True
    else:
        return_objs = False

    for i, (name, expt) in enumerate(experiment_dict.items()):

        if isinstance(expt, AMPSexperiment):
            data = expt.data
        elif isinstance(expt, DataFrame):
            data = expt

        subdata = data[data["frac_labeled"] == frac_labeled]

        (p,) = ax.plot(
            subdata[column1], subdata[column2], ".", alpha=0.75, label=name
        )

        ax.errorbar(
            subdata[column1].mean(),
            subdata[column2].mean(),
            xerr=subdata[column1].std(),
            yerr=subdata[column2].std(),
            fmt="o",
        )

    ax.set_xlabel(f"{column1}")
    ax.set_ylabel(f"{column2}")
    ax.legend()

    if return_objs:
        return fig, ax


def compare_quadratic_fit(experiment_dict, size=(11, 5)):
    fig, ax = plt.subplots(ncols=2, figsize=size)

    for name, expt in experiment_dict.items():
        (p1,) = ax[0].plot(expt.x_P, expt.y_P, ".", alpha=0.5)
        ax[0].errorbar(
            expt.mean_x_P,
            expt.mean_y_P,
            xerr=expt.std_x_P,
            yerr=expt.std_y_P,
            fmt="o",
            color=p1.get_color(),
            ecolor=p1.get_color(),
            elinewidth=1.5,
            capsize=2.5,
            label=name,
        )

        if expt.Pphases is not None:
            ax[0].plot(
                expt.Pphases.fit.x,
                expt.Pphases.fit.y,
                "-",
                lw=2,
                c=p1.get_color(),
            )

        (p2,) = ax[1].plot(expt.x_S, expt.y_S, ".", alpha=0.5)
        ax[1].errorbar(
            expt.mean_x_S,
            expt.mean_y_S,
            xerr=expt.std_x_S,
            yerr=expt.std_y_S,
            fmt="o",
            color=p2.get_color(),
            ecolor=p2.get_color(),
            elinewidth=1.5,
            capsize=2.5,
            label=name,
        )

        if expt.Sphases is not None:
            ax[1].plot(
                expt.Sphases.fit.x,
                expt.Sphases.fit.y,
                "-",
                lw=2,
                c=p2.get_color(),
            )

    ax[0].set_xlabel("P-FLcorr")
    ax[0].set_ylabel("P-SHG")
    ax[1].set_xlabel("S-FLcorr")
    ax[1].set_ylabel("S-SHG")
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()

    return fig, ax


def visualize_AMPS_solution(
    rshg=None, rtpf=None, ax=None, shg_color="darkblue", tpf_color="green"
):

    from matplotlib.lines import Line2D

    if ax is None:
        returnfig = True
        fig, ax = plt.subplots()
    else:
        returnfig = False

    distributions = _shglut["distributions"]
    angles = _shglut["angles"]
    shgratio = _shglut["map"]
    tpfratio = _tpflut["map"]

    if rshg is not None or rshg != np.nan:
        cs_shg = ax.contour(
            distributions,
            angles,
            shgratio,
            levels=[rshg],
            colors=[shg_color],
            linewidths=[2.0],
        )

        ax.clabel(cs_shg, cs_shg.levels)

    if rtpf is not None or rtpf != np.nan:
        cs_tpf = ax.contour(
            distributions,
            angles,
            tpfratio,
            levels=[rtpf],
            colors=[tpf_color],
            linewidths=[2.0],
        )

        ax.clabel(cs_tpf, cs_tpf.levels)

    custom_lines = [
        Line2D([0], [0], color=shg_color, lw=2),
        Line2D([0], [0], color=tpf_color, lw=2),
    ]
    ax.set_xlabel("distribution")
    ax.set_ylabel("angle")
    ax.legend(custom_lines, ["SHG", "TPF"], loc="upper right")

    if returnfig:
        return fig, ax
