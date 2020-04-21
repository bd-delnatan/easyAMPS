"""
A set of modules for convenient file dump to text file (CSV)
"""
import numpy as np
import pdb
from .Text2Prism import (
    PrismFile,
    createPrismColumn,
    assembleColumnPrismTable,
    assembleGroupedPrismTable,
    assembleXYPrismTable,
)


### JUST IN CASE, anticipate for missing data in baseline
def boolean_indexing(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    mask = lens[:, None] > np.arange(lens.max())
    out = np.full(mask.shape, fillval)
    out[mask] = np.concatenate(v)
    return out


### end of JUST IN CASE


def baseline2txt(Plate, generate_Prism_output=False):
    """ Saves baseline data, t=0, to text grouped only by replicates """
    fileprefix = Plate.filename.with_suffix("").name
    filepath = Plate.filename.with_suffix("")
    if not filepath.exists():
        filepath.mkdir(exist_ok=True, parents=True)
    output_file_fmt = str(filepath / (fileprefix + "_baseline_{:s}.txt"))

    if generate_Prism_output:
        prism_output_file = str(filepath / (fileprefix + "_baseline.pzfx"))
        prism_container = PrismFile()

    for i, (ch_name, ch_data) in enumerate(Plate.data):
        conjugate_names = np.unique(Plate.Read_Plate.substance)
        bardata = []
        bar_labels = []
        for conjugate in conjugate_names:
            conjugate_mask = Plate.Read_Plate.substance == conjugate
            for concentration in np.unique(Plate.Read_Plate.concentration):
                conc_mask = Plate.Read_Plate.concentration == concentration
                submask = conjugate_mask & conc_mask
                if submask.sum() > 0:
                    unit = np.unique(Plate.Read_Plate.unit[submask])[0]
                    # get the first timepoint, which is the 'B1' read
                    _data = ch_data[0, submask]
                    bardata.append(_data)
                    bar_labels.append(
                        "{:s} ({:0.2f} {:s})".format(
                            conjugate, concentration, unit
                        )
                    )

        # save to prism file
        if generate_Prism_output:
            col_list = []
            for data, label in zip(bardata, bar_labels):
                # for bargraphs, no X-axis is used, so a 'Column' is used
                col = createPrismColumn(data, title=label, column_type="Y")
                col_list.append(col)
            prism_table = assembleColumnPrismTable(
                col_list,
                name="{:s} channel".format(ch_name),
                ID="Table{:d}".format(i),
            )
            prism_container.append(prism_table)

        # convert bargraph data into numpy array
        bar_data_array = np.array(bardata)

        # unequal Npoints for each conjugate will not form a 2-d array
        if bar_data_array.ndim == 2:
            Npts = bar_data_array.shape[1]
        else:
            # amend this by filling in with nans
            bar_data_array = boolean_indexing(bar_data_array)
            Npts = bar_data_array.shape[1]

        # write raw data to text file. bardata is a list of numpy arrays
        maxtitlelength = max([len(label) for label in bar_labels])
        Ndata = len(bar_labels)
        labelfmt = "{{:>{:d}s}}".format(maxtitlelength)
        datafmt = "{{:>{:d}.1f}}".format(maxtitlelength)
        columnfmt = (labelfmt + ",") * (Ndata - 1) + labelfmt + "\n"
        rowfmt = (datafmt + ",") * (Ndata - 1) + datafmt + "\n"

        with open(output_file_fmt.format(ch_name), "wt") as fhd:
            fhd.write(columnfmt.format(*bar_labels))
            for n in range(Npts):
                fhd.write(rowfmt.format(*bar_data_array[:, n]))

        if generate_Prism_output:
            prism_container.write(prism_output_file)


def save2txt_by_conc(Plate, group_by_conjugates=True, relative_data=False):
    """ Saves all unique experiments into its own text files

    Use in conjunction with the visual/plot dump to help which data points
    need to be further analyzed. This function will create a new folder with
    the same name as the Excel result file. Each channel will also have its own
    folder.

    """
    grouped_experiments = Plate._fetch_unique_experiments(
        return_by_group=True, group_by_conjugates=group_by_conjugates
    )

    # calculate time increment in minutes (the plate outputs in seconds)
    dt_min = Plate.dt_postinject / 60.0

    subfolder_name = ("absolute_signal", "percent_change")[relative_data]
    # path to folder
    filefolder = Plate.filename.with_suffix("") / subfolder_name

    if not filefolder.exists():
        filefolder.mkdir(exist_ok=True, parents=True)

    absolute_suffix = ("absolute", "relative")[relative_data]

    for conjugates, compounds in grouped_experiments.items():
        for compound in compounds:
            expt = Plate.get_unique_experiment(
                conjugates, compound, relative_data=relative_data
            )
            for conjugate in expt.keys():
                data = expt[conjugate]
                for ch_name, ch_data in data.items():
                    # save this data to file, each channel gets its own folder
                    _filename = "{:s}_{:s}_{:s}_{:s}.txt".format(
                        conjugate, compound, ch_name, absolute_suffix
                    )
                    _filefolder = filefolder / "ByConcentration" / ch_name
                    if not _filefolder.exists():
                        _filefolder.mkdir(exist_ok=True, parents=True)
                    _filefullpath = str(_filefolder / _filename)

                    # write to text file
                    with open(_filefullpath, 'wt') as fhd:
                        # str_out contains individual lines of the text file
                        # each line is not line-terminated
                        str_out = []
                        Nreads, Nrep, Nconc = ch_data['data'].shape
                        t_array = np.arange(Nreads) * dt_min

                        # form the header and data output format
                        hdrfmt = (
                            "{:>12s}," + (Nrep - 1) * "{:>12s}," + "{:>12s}"
                        )
                        datfmt = (
                            "{:12e}," + (Nrep - 1) * "{:12.1f}," + "{:12.1f}"
                        )
                        hdrlen = 14 + (Nrep - 1) * 14 + 12
                        concstr = "Conc. ({:s})".format(ch_data['unit'])
                        hdrstr = hdrfmt.format(
                            concstr,
                            *["Rep#{:d}".format(n + 1) for n in range(Nrep)]
                        )

                        # now for each time point
                        for read_id, t in enumerate(t_array):
                            time_id_str = "t = {:0.2f} min".format(t)
                            timehdrfmt = "{{:^{:d}s}}".format(hdrlen)
                            timehdr = timehdrfmt.format(time_id_str)
                            str_out.append(timehdr)
                            str_out.append(hdrstr)
                            # write every concentration and replicate
                            for conc_id in range(Nconc):
                                str_out.append(
                                    datfmt.format(
                                        ch_data['concentrations'][conc_id],
                                        *ch_data['data'][read_id, :, conc_id]
                                    )
                                )
                            str_out.append("\n")
                        # add line-breaks for each end of lines. then write to file
                        fhd.write("\n".join(str_out))


def save2txt_by_time(Plate, group_by_conjugates=True, relative_data=False):
    """ Saves all unique experiments into its own text files

    Use in conjunction with the visual/plot dump to help which data points
    need to be further analyzed. This function will create a new folder with
    the same name as the Excel result file. Each channel will also have its own
    folder.

    """
    grouped_experiments = Plate._fetch_unique_experiments(
        return_by_group=True, group_by_conjugates=group_by_conjugates
    )

    # calculate time increment in minutes (the plate outputs in seconds)
    dt_min = Plate.dt_postinject / 60.0

    subfolder_name = ("absolute_signal", "percent_change")[relative_data]
    # path to folder
    filefolder = Plate.filename.with_suffix("") / subfolder_name

    if not filefolder.exists():
        filefolder.mkdir(exist_ok=True, parents=True)

    absolute_suffix = ("absolute", "relative")[relative_data]

    for conjugate_, compounds in grouped_experiments.items():

        prism_tables = []

        for compound in compounds:
            expt = Plate.get_unique_experiment(
                conjugate_, compound, relative_data=relative_data
            )
            for conjugate in expt.keys():
                data = expt[conjugate]

                for ch_name, ch_data in data.items():
                    # save this data to file, each channel gets its own folder
                    _filename = "{:s}_{:s}_{:s}_{:s}.txt".format(
                        conjugate, compound, ch_name, absolute_suffix
                    )
                    _filefolder = filefolder / "ByTime" / ch_name
                    if not _filefolder.exists():
                        _filefolder.mkdir(exist_ok=True, parents=True)
                    _filefullpath = str(_filefolder / _filename)

                    # write to text file
                    with open(_filefullpath, 'wt') as fhd:
                        # str_out contains individual lines of the text file
                        # each line is not line-terminated
                        str_out = []
                        Nreads, Nrep, Nconc = ch_data['data'].shape
                        t_array = np.arange(Nreads) * dt_min
                        conc_array = ch_data['concentrations']

                        # form the header and data output format
                        hdrfmt = (
                            "{:>12s}," + (Nrep - 1) * "{:>12s}," + "{:>12s}"
                        )
                        datfmt = (
                            "{:12.2f}," + (Nrep - 1) * "{:12.1f}," + "{:12.1f}"
                        )
                        hdrlen = 14 + (Nrep - 1) * 14 + 12
                        timestr = "Time (min)"
                        conjfmt = "{{:^{:d}s}}".format(hdrlen)
                        conjstr = conjfmt.format(conjugate)
                        hdrstr = hdrfmt.format(
                            timestr,
                            *["Rep#{:d}".format(n + 1) for n in range(Nrep)]
                        )
                        str_out.append(conjstr)

                        for conc_id, c in enumerate(conc_array):
                            conc_id_str = "[{:s}] = {:0.4f} {:s}".format(
                                compound, c, ch_data['unit']
                            )
                            # place the title at the center of columns
                            conchdrfmt = "{{:^{:d}s}}".format(hdrlen)
                            conchdr = conchdrfmt.format(conc_id_str)
                            str_out.append(conchdr)
                            str_out.append(hdrstr)
                            # write out every time point and replicate
                            for t_id in range(Nreads):
                                str_out.append(
                                    datfmt.format(
                                        t_array[t_id],
                                        *ch_data['data'][t_id, :, conc_id]
                                    )
                                )
                            str_out.append("\n")

                        fhd.write("\n".join(str_out))
