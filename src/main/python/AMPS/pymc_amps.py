import pymc3 as pm
import theano.tensor as tt
import numpy as np
import pickle
from pathlib import Path
from .ops import theano_P_SHG, theano_S_SHG, theano_P_TPF, theano_S_TPF
from .functions import _tpfratio


def save_inference(output_dir, output_fn, model, traces):

    outdir = Path(output_dir)

    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)

    out_fullfn = f"{str(outdir / output_fn)}.pickle"

    with open(out_fullfn, "wb") as fhd:
        pickle.dump({"model": model, "traces": traces}, fhd)
        print(f"PyMC3 inference has been saved to {out_fullfn}")


def load_inference(pickle_file):

    with open(pickle_file, "rb") as fhd:

        inference = pickle.load(fhd)

    return inference["model"], inference["traces"]


def form_AMPS_model_01(data):
    """ full 4-channel measurement analysis """

    f, idx = np.unique(
        data.query("frac_labeled > 0")["frac_labeled"].values,
        return_inverse=True,
    )

    labeled = data.query("frac_labeled > 0")
    unlabeled = data.query("frac_labeled == 0")

    pshg_obs = labeled["P-SHG"].values
    sshg_obs = labeled["S-SHG"].values
    ptpf_obs = labeled["P-FLcorr"].values
    stpf_obs = labeled["S-FLcorr"].values

    background_pshg_mean, background_pshg_sd = unlabeled["P-SHG"].agg(
        ["mean", "std"]
    )
    background_sshg_mean, background_sshg_sd = unlabeled["S-SHG"].agg(
        ["mean", "std"]
    )

    #  estimate for signal by groups, 'frac_labeled'
    labelgrp = labeled.groupby("frac_labeled")
    pshg_sd = labelgrp["P-SHG"].agg("std").values
    sshg_sd = labelgrp["S-SHG"].agg("std").values
    ptpf_sd = labelgrp["P-FLcorr"].agg("std").values
    stpf_sd = labelgrp["S-FLcorr"].agg("std").values

    model = pm.Model()

    with model:
        # bounded normal distribution
        BoundedPhase = pm.Bound(pm.Normal, lower=0.0, upper=np.deg2rad(180.0))
        NonNegNormal = pm.Bound(pm.Normal, lower=0.0)

        angle = pm.Uniform(
            "angle",
            lower=np.deg2rad(5.0),
            upper=np.deg2rad(90.0),
            testval=np.deg2rad(60.0),
        )
        distribution = pm.Uniform(
            "distribution",
            lower=np.deg2rad(2.0),
            upper=np.deg2rad(70.0),
            testval=np.deg2rad(25.0),
        )

        # Scaling constants
        preC = pm.Beta("c_", alpha=2, beta=2, testval=0.5)
        preD = pm.Beta("d_", alpha=2, beta=2, testval=0.5)

        # C = NonNegNormal("C", mu=1.4e5, sd=4e4, testval=1.2e5)
        # D = NonNegNormal("D", mu=1.2e5, sd=2e4, testval=1.0e5)
        C = pm.Deterministic("C", preC * 5e5)
        D = pm.Deterministic("D", preD * 5e5)

        varphi_P = pm.Uniform("phase_P", lower=0.0, upper=np.pi)
        varphi_S = pm.Uniform("phase_S", lower=0.0, upper=np.pi)

        # calculate the forward model
        PSHG_BG = NonNegNormal(
            "PSHG_BG", mu=background_pshg_mean, sd=background_pshg_sd
        )
        SSHG_BG = NonNegNormal(
            "SSHG_BG", mu=background_sshg_mean, sd=background_sshg_sd
        )

        fadj = pm.Normal("f", mu=f, sd=0.1, shape=len(f))

        PSHG_probe = C * fadj ** 2 * 1.88 * theano_P_SHG(angle, distribution)
        SSHG_probe = C * fadj ** 2 * theano_S_SHG(angle, distribution)

        Ippp = (
            PSHG_BG
            + PSHG_probe
            + 2 * tt.sqrt(PSHG_BG) * tt.sqrt(PSHG_probe) * tt.cos(varphi_P)
        )
        Ipss = (
            SSHG_BG
            + SSHG_probe
            + 2 * tt.sqrt(SSHG_BG) * tt.sqrt(SSHG_probe) * tt.cos(varphi_S)
        )

        Fpp = D * fadj * 1.88 * theano_P_TPF(angle, distribution)
        Fss = D * fadj * theano_S_TPF(angle, distribution)

        # express likelihood
        ll_Ippp = pm.Normal(
            "Ippp", mu=Ippp[idx], sd=pshg_sd[idx], observed=pshg_obs
        )
        ll_Ipss = pm.Normal(
            "Ipss", mu=Ipss[idx], sd=sshg_sd[idx], observed=sshg_obs
        )
        ll_Fpp = pm.Normal(
            "Fpp", mu=Fpp[idx], sd=ptpf_sd[idx], observed=ptpf_obs
        )
        ll_Fss = pm.Normal(
            "Fss", mu=Fss[idx], sd=stpf_sd[idx], observed=stpf_obs
        )

        # express in degrees for tracking
        angle_deg = pm.Deterministic("angle_degree", angle * 180.0 / np.pi)
        distribution_deg = pm.Deterministic(
            "distribution_degree", distribution * 180.0 / np.pi
        )
        varphi_P_deg = pm.Deterministic(
            "phase_P_degree", varphi_P * 180.0 / np.pi
        )
        varphi_S_deg = pm.Deterministic(
            "phase_S_degree", varphi_S * 180.0 / np.pi
        )

        return model


def form_sequential_AMPS_model(data):

    f, idx = np.unique(
        data.query("frac_labeled > 0")["frac_labeled"].values,
        return_inverse=True,
    )

    labeled = data.query("frac_labeled > 0")
    unlabeled = data.query("frac_labeled == 0")

    pshg_obs = labeled["P-SHG"].values
    sshg_obs = labeled["S-SHG"].values
    ptpf_obs = labeled["P-FLcorr"].values
    stpf_obs = labeled["S-FLcorr"].values

    background_pshg_mean, background_pshg_sd = unlabeled["P-SHG"].agg(
        ["mean", "std"]
    )
    background_sshg_mean, background_sshg_sd = unlabeled["S-SHG"].agg(
        ["mean", "std"]
    )

    #  estimate for signal by groups, 'frac_labeled'
    labelgrp = labeled.groupby("frac_labeled")
    pshg_sd = labelgrp["P-SHG"].agg("std").values
    sshg_sd = labelgrp["S-SHG"].agg("std").values
    ptpf_sd = labelgrp["P-FLcorr"].agg("std").values
    stpf_sd = labelgrp["S-FLcorr"].agg("std").values

    tpfratio = labeled.loc[labeled["frac_labeled"] == 1.0, "TPFratio"].mean()
    tpfratio_sd = labeled.loc[labeled["frac_labeled"] == 1.0, "TPFratio"].std()

    angles_vec = np.deg2rad(np.linspace(0.0, 90.0, num=150))
    dist_vec = np.deg2rad(np.linspace(2.0, 70.0, num=100))
    SIGMA, MU = np.meshgrid(dist_vec, angles_vec, indexing="ij")
    rtpfmap = _tpfratio(MU, SIGMA)
    floglik = np.exp(-(((rtpfmap - tpfratio) / tpfratio_sd) ** 2) / 2.0)
    π_angles = floglik.sum(axis=0)
    π_distributions = floglik.sum(axis=1)
    prior_angles = π_angles / (π_angles * (angles_vec[1] - angles_vec[0])).sum()
    prior_distributions = (
        π_distributions / (π_distributions * (dist_vec[1] - dist_vec[0])).sum()
    )

    model = pm.Model()

    with model:

        NonNegNormal = pm.Bound(pm.Normal, lower=0.0)

        angle = pm.Interpolated("angle", angles_vec, prior_angles)
        distribution = pm.Interpolated(
            "distribution", dist_vec, prior_distributions
        )

        # Scaling constants
        preC = pm.Beta("c_", alpha=2, beta=2, testval=0.5)
        preD = pm.Beta("d_", alpha=2, beta=2, testval=0.5)

        C = pm.Deterministic("C", preC * 1e6)
        D = pm.Deterministic("D", preD * 5e5)

        varphi_P = pm.Uniform("phase_P", lower=0.0, upper=np.pi)
        varphi_S = pm.Uniform("phase_S", lower=0.0, upper=np.pi)

        # calculate the forward model
        PSHG_BG = NonNegNormal(
            "PSHG_BG", mu=background_pshg_mean, sd=background_pshg_sd
        )
        SSHG_BG = NonNegNormal(
            "SSHG_BG", mu=background_sshg_mean, sd=background_sshg_sd
        )

        fadj = pm.Normal("f", mu=f, sd=0.1, shape=len(f))

        PSHG_probe = C * fadj ** 2 * 1.88 * theano_P_SHG(angle, distribution)
        SSHG_probe = C * fadj ** 2 * theano_S_SHG(angle, distribution)

        Ippp = (
            PSHG_BG
            + PSHG_probe
            + 2 * tt.sqrt(PSHG_BG) * tt.sqrt(PSHG_probe) * tt.cos(varphi_P)
        )
        Ipss = (
            SSHG_BG
            + SSHG_probe
            + 2 * tt.sqrt(SSHG_BG) * tt.sqrt(SSHG_probe) * tt.cos(varphi_S)
        )

        Fpp = D * fadj * 1.88 * theano_P_TPF(angle, distribution)
        Fss = D * fadj * theano_S_TPF(angle, distribution)

        # express likelihood
        ll_Ippp = pm.Normal(
            "Ippp", mu=Ippp[idx], sd=pshg_sd[idx], observed=pshg_obs
        )
        ll_Ipss = pm.Normal(
            "Ipss", mu=Ipss[idx], sd=sshg_sd[idx], observed=sshg_obs
        )
        ll_Fpp = pm.Normal(
            "Fpp", mu=Fpp[idx], sd=ptpf_sd[idx], observed=ptpf_obs
        )
        ll_Fss = pm.Normal(
            "Fss", mu=Fss[idx], sd=stpf_sd[idx], observed=stpf_obs
        )

        # express in degrees for tracking
        angle_deg = pm.Deterministic("angle_degree", angle * 180.0 / np.pi)
        distribution_deg = pm.Deterministic(
            "distribution_degree", distribution * 180.0 / np.pi
        )
        varphi_P_deg = pm.Deterministic(
            "phase_P_degree", varphi_P * 180.0 / np.pi
        )
        varphi_S_deg = pm.Deterministic(
            "phase_S_degree", varphi_S * 180.0 / np.pi
        )

    return model


def form_AMPS_model_FL(data):
    """ AMPS model using fluorescence alone """

    f, idx = np.unique(
        data.query("frac_labeled > 0")["frac_labeled"].values,
        return_inverse=True,
    )

    labeled = data.query("frac_labeled > 0")

    ptpf_obs = labeled["P-FLcorr"].values
    stpf_obs = labeled["S-FLcorr"].values

    #  estimate for signal by groups, 'frac_labeled'
    labelgrp = labeled.groupby("frac_labeled")
    ptpf_sd = labelgrp["P-FLcorr"].agg("std").values
    stpf_sd = labelgrp["S-FLcorr"].agg("std").values

    model = pm.Model()

    with model:

        angle = pm.Uniform(
            "angle",
            lower=np.deg2rad(5.0),
            upper=np.deg2rad(90.0),
            testval=np.deg2rad(60.0),
        )
        distribution = pm.Uniform(
            "distribution",
            lower=np.deg2rad(2.0),
            upper=np.deg2rad(70.0),
            testval=np.deg2rad(25.0),
        )

        # Scaling constants
        preD = pm.Beta("d_", alpha=2, beta=2, testval=0.5)
        D = pm.Deterministic("D", preD * 5e5)

        fadj = pm.Normal("f", mu=f, sd=0.1, shape=len(f))

        Fpp = D * fadj * 1.88 * theano_P_TPF(angle, distribution)
        Fss = D * fadj * theano_S_TPF(angle, distribution)

        # express likelihood
        ll_Fpp = pm.Normal(
            "Fpp", mu=Fpp[idx], sd=ptpf_sd[idx], observed=ptpf_obs
        )
        ll_Fss = pm.Normal(
            "Fss", mu=Fss[idx], sd=stpf_sd[idx], observed=stpf_obs
        )

        # express in degrees for tracking
        angle_deg = pm.Deterministic("angle_degree", angle * 180.0 / np.pi)
        distribution_deg = pm.Deterministic(
            "distribution_degree", distribution * 180.0 / np.pi
        )

    return model
