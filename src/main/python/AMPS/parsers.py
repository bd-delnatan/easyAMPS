"""
modules containing utility functions used in parsing AMPS phase-determination
experiments.

"""

import numpy as np
import pandas as pd
from itertools import product
import re

# row look up table
row2id = {letter: index + 1 for index, letter in enumerate("ABCDEFGHIJKLMNOP")}
id2row = {index: letter for letter, index in row2id.items()}


def form_dilution_block(
    columns, origin, dilution_factor=2.0 / 3.0, transpose=False
):
    """ returns plate map for dilution series

    Args:
        columns (list of columns): containing dilution series sequence from 1,..,N.
        origin (str): well coordinates denoting origin of block (e.g. "A1")
        dilution_factor (float): dilution factor
        transpose (bool): if True, then input columns will be transposed,
        assuming that input has dimensions [columns, rows].

    Returns:
        DataFrame containing columns "Well coordinates" and "frac_labeled"
    """

    rowinit = origin[0].upper()
    colinit = int(origin[1:])

    if transpose:
        block = np.array(columns).T
    else:
        block = np.array(columns)

    # convert 'blank' entries into nan
    block[block == 'blank'] = np.nan

    # coerce into float
    block = block.astype(np.float64)

    # comput dilution factor
    block = dilution_factor ** (block - 1)

    # replace lowest dilution factor with 0
    block[block == np.nanmin(block)] = 0.0

    # round to the nearest 3-digit
    block = np.around(block, decimals=3)

    Nrows, Ncols = block.shape

    column_names = [colinit + i for i in range(Ncols)]
    row_names = [id2row[row2id[rowinit] + i] for i in range(Nrows)]

    dilution_map = [
        (f"{w[0]}{w[1]}", v)
        for w, v in zip(product(row_names, column_names), block.ravel())
    ]

    df = pd.DataFrame(
        dilution_map, columns=["Well coordinates", "frac_labeled"]
    )

    return df


def get_series(dataframe, series):

    subset = dataframe[
        dataframe["Well coordinates"].isin(series["Well coordinates"].unique())
    ].copy()

    subset = subset.merge(series)

    return subset


def process_experiment(
    data, config, experiment_name, save_data=False, transpose_block=True
):
    datadict = {}

    for experiment, exptconfig in config[experiment_name].items():

        origins = exptconfig["origins"].split(",")
        pattern = config["patterns"][exptconfig["pattern"]]

        # if there are wells to be excluded (e.g. bad wells)
        if "exclude" in exptconfig.keys():
            wells_to_exclude = exptconfig["exclude"].split(",")
            data = data[~data["Well coordinates"].isin(wells_to_exclude)].copy()

        dfs = []

        # if multiple origins were given, merge them
        if len(origins) > 1:
            for origin in origins:
                dilution_map = form_dilution_block(
                    pattern, origin, transpose=transpose_block
                )
                df = get_series(data, dilution_map)
                # zero-shift fluorescence subtract by unlabeled protein wells
                pfl_bg = df.query("frac_labeled == 0")["P-FLcorr"].mean()
                sfl_bg = df.query("frac_labeled == 0")["S-FLcorr"].mean()
                df["P-FLcorr"] -= pfl_bg
                df["S-FLcorr"] -= sfl_bg
                df.loc[df["frac_labeled"] > 0, "TPFratio"] = (
                    df.loc[df["frac_labeled"] > 0, "P-FLcorr"]
                    / df.loc[df["frac_labeled"] > 0, "S-FLcorr"]
                )
                dfs.append(df)
            datadict[experiment] = pd.concat(dfs, axis=0)

        # otherwise, just read the data
        elif len(origins) == 1:

            dilution_map = form_dilution_block(
                pattern, origins[0], transpose=transpose_block
            )
            df = get_series(data, dilution_map)
            pfl_bg = df.query("frac_labeled == 0")["P-FLcorr"].mean()
            sfl_bg = df.query("frac_labeled == 0")["S-FLcorr"].mean()
            df["P-FLcorr"] -= pfl_bg
            df["S-FLcorr"] -= sfl_bg
            df.loc[df["frac_labeled"] > 0, "TPFratio"] = (
                df.loc[df["frac_labeled"] > 0, "P-FLcorr"]
                / df.loc[df["frac_labeled"] > 0, "S-FLcorr"]
            )
            datadict[experiment] = df

    return datadict


def validate_well_coordinates(wclist):

    # will contain a list of bool
    tflist = []
    # wells that are invalid
    invalidlist = []

    wellptn = re.compile("[a-pA-P][0-9]{1,2}")

    for well in wclist:
        m = wellptn.match(well)
        # check validity of well coordinate format
        ok = bool(m)

        if ok:
            # then check if well fits within 384-well
            rowchar = well[0]
            colnum = int(well[1:])

            if 0 < colnum < 25:
                tflist.append(True)
            else:
                tflist.append(False)
                invalidlist.append(well)

        elif not ok:
            tflist.append(False)
            invalidlist.append(well)

    return all(tflist), invalidlist
