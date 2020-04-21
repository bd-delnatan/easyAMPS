"""
Plate visualization for quick analysis of SHG experiments.


version history

5/22/2019 - DE

    * a generic multi-page PDF class for handling multi-page PDF generation
    * basic functionalities implemented to generate a grid of plots to a
      multi-page PDF for every unique experiment (a conjugate + compound pair)

8/30/2019 - DE
    * This has now been deprecated. Kept only for backwards compatibility

"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from math import ceil
from matplotlib.backends.backend_pdf import PdfPages

fontsettings = {"size": 6}
mpl.rc("font", **fontsettings)
# axis spine customization (for L-shaped axes frame)
mpl.rcParams["axes.spines.left"] = True
mpl.rcParams["axes.spines.bottom"] = True
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False

# for setting colors
# winter_cmap = lambda n: plt.cm.winter(np.linspace(0, 0.9, n))

# CONSTANTS
LANDSCAPE = (11, 8.5)
PORTRAIT = (8.5, 11)


class MultipagePDF:
    """A class to manage plotting across multi-page PDF

        Opens a PDF file stream for convenient multi-page PDF for plots arranged
        in a grid. Uses matplotlib.backends.backend_pdf.PdfPages.

        Args:
            filename(str): the filename of PDF file to be created.
            nrows(int): number of rows for individual plots.
            ncols(int): number of columns for individual plots.
            orientation(str): either 'landscape' or 'portrait' or a tuple of
                page size in inches (width, height).
            first_page_title(str): the title for the first page.
            orientation(str): "landscape" or "portrait". This parameter
                determines the paper orientation. The "portrait" orientation
                has size of 8.5x11-inch.

        Example:

        .. code-block:: python

            import numpy as np
            import matplotlib.pyplot as plt
            from PyBiodesy.PlateVisualization import MultipagePDF
safari

            Nplots = 26
            Ndata  = 10
            x = np.arange(Ndata) * 0.25

            with MultipagePDF("./testpage.pdf",
                              nrows=5, ncols=4,
                             first_page_title="My first page") as mpdf:
                for n in range(Nplots):
                    y = np.random.randn(Ndata)
                    ax = mpdf.new_axis()
                    ax.plot(x, y, 'o-')
                    ax.set_xlabel("X-axis")
                    ax.set_ylabel("Y-axis")
                    ax.set_title("Plot #{:d}".format(n+1))

                mpdf.start_new_page(page_title="A new page")

                # start new plot
                for n in range(12):
                    y = np.random.randn(Ndata)
                    ax = mpdf.new_axis()
                    ax.plot(x, y, 'ro-')
                    ax.set_xlabel("X-axis")
                    ax.set_ylabel("Y-axis")
                    ax.set_title("Red Plot #{:d}".format(n+1))

    """

    def __init__(
        self,
        filename,
        nrows=3,
        ncols=3,
        first_page_title=None,
        orientation="landscape",
    ):

        self.filename = filename
        self.Nrows = nrows
        self.Ncols = ncols
        self.Nplots_per_page = self.Nrows * self.Ncols
        self.orientation = "landscape"
        self.LANDSCAPE = (11, 8.5)
        self.PORTRAIT = (8.5, 11)
        self.figsize = ((8.5, 11), (11, 8.5))[orientation == "landscape"]
        self.plotcount = 0
        self.initialized = False
        self.pdfhandle = PdfPages(filename)
        self.pagenum_rotation = "vertical"
        self.pagenum_x = 0.3
        self.pagenum_y = 0.6
        self.pagenum_fs = 6
        self.pagenum_color = "#808080"
        #
        plt.subplots_adjust(top=0.92)

        # create a matplotlib axis
        self.start_new_page(
            Nrows=self.Nrows,
            Ncols=self.Ncols,
            figsize=self.figsize,
            page_title=first_page_title,
        )

    def start_new_page(
        self, Nrows=None, Ncols=None, figsize=None, page_title=None
    ):
        """ start a new PDF page

        The default parameters are set in

        Args:
            Nrows(int): number of rows in new page.
            Ncols(int): number of columns in new page.
            figsize(tuple of ints): the (width,height) of new page in inches.
            page_title(str): the title of the new page.

        """
        if Nrows is not None:
            self.Nrows = Nrows
        if Ncols is not None:
            self.Ncols = Ncols
        if figsize is not None:
            self.figsize = figsize

        # recompute Nplots per page for the new page
        self.Nplots_per_page = self.Nrows * self.Ncols

        if not self.initialized:
            # if this is the first page, there are no plots/pages to save
            # so start a new one with new parameters
            # reset plot counter
            self.plotcount = 0
            self.initialized = True

        elif self.initialized:
            # otherwise, save the plots to current page and start a new one
            # with the parameters specified
            self.cleanup()
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            self.pdfhandle.savefig()
            plt.close()

            # reset plot counter
            self.plotcount = 0

        self.fig, self.axes_array = plt.subplots(
            nrows=self.Nrows, ncols=self.Ncols, figsize=self.figsize
        )

        x = float(self.figsize[0] - self.pagenum_x) / float(self.figsize[0])
        y = float(self.figsize[1] - self.pagenum_y) / float(self.figsize[1])
        self.fig.text(
            x,
            y,
            "Page {:d}".format(self.pdfhandle.get_pagecount() + 1),
            fontsize=self.pagenum_fs,
            rotation=self.pagenum_rotation,
            color=self.pagenum_color,
        )
        if page_title is not None:
            self.fig.suptitle(page_title)

    def next_page(self, page_title=None):
        """ create a new page without resetting counter """
        self.fig.set_tight_layout(None)
        plt.subplots_adjust(top=0.92)
        plt.tight_layout()
        self.pdfhandle.savefig()
        plt.close(self.fig)

        self.fig, self.axes_array = plt.subplots(
            nrows=self.Nrows, ncols=self.Ncols, figsize=self.figsize
        )

        # position of page number text, float [0,1]
        x = float(self.figsize[0] - self.pagenum_x) / float(self.figsize[0])
        y = float(self.figsize[1] - self.pagenum_y) / float(self.figsize[1])

        # add page number
        self.fig.text(
            x,
            y,
            "Page {:d}".format(self.pdfhandle.get_pagecount() + 1),
            fontsize=self.pagenum_fs,
            rotation=self.pagenum_rotation,
            color=self.pagenum_color,
        )

        if page_title is not None:
            self.fig.suptitle(page_title)

    def new_axis(self):
        """ returns the next Axes object and increment plot counter """
        if self.plotcount != 0 and self.plotcount % self.Nplots_per_page == 0:
            # if we have reached the end of current page, create a new one
            # thus populating new self.axes_array
            self.next_page()
            n = self.plotcount % self.Nplots_per_page
            row = n // self.Ncols
            col = n % self.Ncols
            self.plotcount += 1

            # for a vector of axes_array, we don't use array indexing
            if type(self.axes_array) == np.ndarray:
                if self.axes_array.ndim == 1:
                    return self.axes_array[n]
                else:
                    return self.axes_array[row, col]
            elif issubclass(type(self.axes_array), mpl.axes.SubplotBase):
                return self.axes_array

        else:
            n = self.plotcount % self.Nplots_per_page
            row = n // self.Ncols
            col = n % self.Ncols

            self.plotcount += 1

            # for a vector of axes_array, we don't use array indexing
            if type(self.axes_array) == np.ndarray:
                if self.axes_array.ndim == 1:
                    return self.axes_array[n]
                elif self.axes_array.ndim == 2:
                    return self.axes_array[row, col]
            elif issubclass(type(self.axes_array), mpl.axes.SubplotBase):
                return self.axes_array

    def cleanup(self):
        """ Hides the remaining unused Axes """
        Nremainder = self.plotcount % self.Nplots_per_page
        if Nremainder != 0:
            Ncleanup = self.Nplots_per_page - Nremainder
            if Ncleanup != 0:
                # go through each axis and set them to be 'invisible'
                for n in range(Ncleanup):
                    ax = self.new_axis()
                    ax.set_axis_off()

    def close(self):
        """ do some cleanup and close the PDF file stream """
        self.cleanup()
        print("cleaning up")
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        self.pdfhandle.savefig()
        plt.close()

        # close pdf file stream
        self.pdfhandle.close()

    def __enter__(self):
        """ For using a context manager """
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """ Upon exit, this gets called """
        self.close()
        print("{:s} is closed".format(self.filename))


def visualize_plate_data(
    array,
    title="Plate view",
    axis_handle=None,
    colorbar_orientation="vertical",
    **kwargs
):
    """ do the augmented matplotlib.pyplot.imshow for 384-well plate

        The extra arguments are passed onto pyplot.imshow()

        """
    if axis_handle is None:
        _view = plt.imshow
        _title = plt.title
        _hlines = plt.hlines
        _vlines = plt.vlines
        _yticks = lambda y, ylabels, fontdict: plt.yticks(
            y, ylabels, **fontdict
        )
        _xticks = lambda x, xlabels, fontdict: plt.xticks(
            x, xlabels, **fontdict
        )
    else:
        _view = axis_handle.imshow
        _title = axis_handle.set_title
        _hlines = axis_handle.hlines
        _vlines = axis_handle.vlines
        _xticks = lambda x, xlabels, **kwargs: [
            axis_handle.set_xticks(x),
            axis_handle.set_xticklabels(xlabels, **kwargs),
        ]
        _yticks = lambda y, ylabels, **kwargs: [
            axis_handle.set_yticks(y),
            axis_handle.set_yticklabels(ylabels, **kwargs),
        ]

    mappable = _view(array, **kwargs)
    _title(title)
    _hlines(
        y=np.arange(1 + self.height) - 0.5,
        xmin=-0.5,
        xmax=self.width - 0.5,
        colors="#999999",
    )
    _vlines(
        x=np.arange(1 + self.width) - 0.5,
        ymin=-0.5,
        ymax=self.height - 0.5,
        colors="#999999",
    )
    _yticks(
        np.arange(len(self.row_labels)),
        self.row_labels,
        fontdict={"fontsize": "x-small"},
    )
    _xticks(
        np.arange(len(self.col_labels)),
        self.col_labels,
        fontdict={"fontsize": "x-small"},
    )

    if axis_handle is None:
        plt.colorbar(orientation=colorbar_orientation, pad=0.15)
    else:
        plt.colorbar(
            mappable, orientation=colorbar_orientation, pad=0.15, ax=axis_handle
        )


def visualize_baseline(Plate):
    """ Saves the signal-check and baseline plots  """
    filefolder = Plate.filename.with_suffix("")
    if not filefolder.exists():
        filefolder.mkdir(exist_ok=True, parents=True)

    # prepare output filename formatting
    fileprefix = Plate.filename.with_suffix("").name
    # the filename format
    filenamefmt = "{:s}_signalcheck.pdf".format(fileprefix)
    targetfilefmt = str(filefolder / filenamefmt)
    pdffilename = targetfilefmt.format(fileprefix)

    # the number of raw data channels
    Nimg = len(Plate.rawreads)

    if Nimg < 3:
        Ncols = 1
        Nrows = 2
    elif Nimg >= 3:
        Ncols = 2
        Nrows = ceil(Nimg / Ncols)

    with MultipagePDF(
        pdffilename,
        nrows=Nrows,
        ncols=Ncols,
        first_page_title="Plate CV of raw data (last read)",
        orientation="portrait",
    ) as mpdf:

        for ch_name, raw_array in Plate.rawreads.items():
            if "PD" in ch_name:
                vmax = 0.5
            else:
                vmax = None
            ax = mpdf.new_axis()
            Plate.visualize_plate_data(
                np.std(raw_array[-1, ...], axis=-1)
                / np.median(raw_array[-1, ...], axis=-1),
                title="{:s} StDev/Median".format(ch_name),
                axis_handle=ax,
                vmin=0,
                vmax=vmax,
            )

        mpdf.start_new_page(Nrows=1, Ncols=1, figsize=mpdf.PORTRAIT)

        for ch_name, ch_data in Plate.data:
            # loop through each channel in plate data
            # gather data for plotting
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
                        # get the zeroeth timepoint, the 'B1' read
                        _data = ch_data[0, submask]
                        bardata.append(_data)
                        bar_labels.append(
                            "{:s} ({:0.1f} {:s})".format(
                                conjugate, concentration, unit
                            )
                        )
            # then plot a box plot for each channel
            ax = mpdf.new_axis()
            ax.boxplot(bardata, labels=bar_labels)
            if len(bar_labels) > 4:
                for tick in ax.get_xticklabels():
                    tick.set_rotation(90)
            ax.set_ylabel("Intensity ({:s})".format(ch_name))
            ax.set_title("Baseline read ({:s})".format(ch_name))


def visualize_by_time(Plate, relative_data=False, global_scale=False):
    """ Kinetic summary of the experiment from the whole plate """
    subfolder_name = ("absolute_signal", "percent_change")[relative_data]
    # path to folder
    filefolder = Plate.filename.with_suffix("") / subfolder_name / "ByTime"
    if not filefolder.exists():
        filefolder.mkdir(exist_ok=True, parents=True)

    # define time increment in minutes
    dt_min = Plate.dt_postinject / 60.0

    # prepare output filename formatting
    fileprefix = Plate.filename.with_suffix("").name
    absolute_suffix = ("absolute", "relative")[relative_data]

    # the filename format (prefix_suffix_modality.pdf)
    filenamefmt = "{:s}_{:s}_{{:s}}.pdf".format(fileprefix, absolute_suffix)
    # the full target file includes the folder path
    targetfilefmt = str(filefolder / filenamefmt)

    for ch_name in Plate.data.channel_names:

        if global_scale and relative_data:
            global_min = Plate.relative_data.__dict__[ch_name].min()
            global_max = Plate.relative_data.__dict__[ch_name].max()
        elif global_scale and not relative_data:
            global_min = Plate.data.__dict__[ch_name].min()
            global_max = Plate.data.__dict__[ch_name].max()

        grouped_experiments = Plate._fetch_unique_experiments()

        with MultipagePDF(
            targetfilefmt.format(ch_name), nrows=3, ncols=4
        ) as mpdf:
            for conjugate, compound in grouped_experiments:
                # expt is a 'unique' experiment dictionary
                # which contains multiple concentrations for
                # the same conjugate.
                expt = Plate.get_unique_experiment(
                    conjugate, compound, relative_data=relative_data
                )
                for unique_conjugate, unique_expt in expt.items():
                    ch_data = unique_expt[ch_name]
                    _unit = ch_data["unit"]

                    y_intensity_label = ("{:s} intensity", "%Δ{:s}")[
                        relative_data
                    ]

                    # form title
                    title_str = conjugate.split("+")
                    title_fmt = "\n".join(
                        ["{:s} +" for _ in range(len(title_str))]
                    )

                    # get data array
                    Nreads, Nrep, Nconcs = ch_data["data"].shape

                    # save this visual summary as its own file per channel
                    t_array = np.arange(Nreads) * dt_min

                    ax = mpdf.new_axis()
                    ax.set_xlabel("Time, minutes")
                    ax.set_ylabel(y_intensity_label.format(ch_name))
                    ax.set_title(
                        (title_fmt + "\n{:s}").format(*title_str, compound)
                    )

                    markercolors = winter_cmap(Nconcs)

                    for j in range(Nconcs):
                        y_data = ch_data["data"][:, :, j].mean(axis=1)
                        y_err = ch_data["data"][:, :, j].std(axis=1)
                        ax.errorbar(
                            t_array,
                            y_data,
                            yerr=y_err,
                            fmt="o-",
                            color=markercolors[j],
                            label="{:0.2f} {:s}".format(
                                ch_data["concentrations"][j], _unit
                            ),
                        )

                        if global_scale:
                            ax.set_ylim(bottom=global_min, top=global_max)


def visualize_by_concentration(Plate, relative_data=False, global_scale=False):
    """ Kinetic summary of the experiment from the whole plate

    Since only one file is generated per channel, we don't have the
    """
    subfolder_name = ("absolute_signal", "percent_change")[relative_data]
    # path to folder
    filefolder = (
        Plate.filename.with_suffix("") / subfolder_name / "ByConcentration"
    )
    if not filefolder.exists():
        filefolder.mkdir(exist_ok=True, parents=True)

    # define time increment in minutes
    dt_min = Plate.dt_postinject / 60.0
    time_unit = "min"
    # prepare output filename formatting
    fileprefix = Plate.filename.with_suffix("").name
    absolute_suffix = ("absolute", "relative")[relative_data]

    # the filename format
    filenamefmt = "{:s}_{:s}_{{:s}}.pdf".format(fileprefix, absolute_suffix)
    targetfilefmt = str(filefolder / filenamefmt)

    for ch_name in Plate.data.channel_names:
        if global_scale and relative_data:
            global_min = Plate.relative_data.__dict__[ch_name].min()
            global_max = Plate.relative_data.__dict__[ch_name].max()
        elif global_scale and not relative_data:
            global_min = Plate.data.__dict__[ch_name].min()
            global_max = Plate.data.__dict__[ch_name].max()

        grouped_experiments = Plate._fetch_unique_experiments()
        with MultipagePDF(
            targetfilefmt.format(ch_name), nrows=3, ncols=4
        ) as mpdf:
            for conjugate, compound in grouped_experiments:
                # expt is a 'unique' experiment dictionary
                # which contains multiple concentrations for
                # the same conjugate.
                expt = Plate.get_unique_experiment(
                    conjugate, compound, relative_data=relative_data
                )
                for unique_conjugate, unique_expt in expt.items():
                    ch_data = unique_expt[ch_name]
                    _unit = ch_data["unit"]

                    y_intensity_label = ("{:s} intensity", "%Δ{:s}")[
                        relative_data
                    ]

                    # form title
                    title_str = conjugate.split("+")
                    title_fmt = "\n".join(
                        ["{:s} +" for _ in range(len(title_str))]
                    )

                    # get data array
                    Nreads, Nrep, Nconcs = ch_data["data"].shape

                    # save this visual summary as its own file per channel
                    ax = mpdf.new_axis()
                    ax.set_xlabel("[{:s}], {:s}".format(compound, _unit))
                    ax.set_ylabel(y_intensity_label.format(ch_name))
                    ax.set_title(
                        (title_fmt + "\n{:s}").format(*title_str, compound)
                    )

                    x_data = ch_data["concentrations"]
                    t_array = np.arange(Nreads) * dt_min

                    for j in range(Nreads):
                        y_data = ch_data["data"][j, :, :].mean(axis=0)
                        y_err = ch_data["data"][j, :, :].std(axis=0)
                        if j > 0:
                            ax.errorbar(
                                x_data,
                                y_data,
                                yerr=y_err,
                                fmt="o-",
                                label="{:0.2f} {:s}".format(
                                    t_array[j], time_unit
                                ),
                            )
                        else:
                            ax.plot(x_data, y_data, "--", c="#808080")

                    ax.set_xscale("log")

                    if global_scale:
                        ax.set_ylim(bottom=global_min, top=global_max)


def general_visualize_by_time(
    Plate, y_data="SHGratio", global_scale=False, ylims=None
):
    """ A general summary of the experiment from the whole plate

    The operator str gives you flexibility for what would be used to be
    on the Y-axis.

    """
    subfolder_name = y_data

    # path to folder
    filefolder = Plate.filename.with_suffix("") / subfolder_name / "ByTime"
    if not filefolder.exists():
        filefolder.mkdir(exist_ok=True, parents=True)

    # define time increment in minutes
    dt_min = Plate.dt_postinject / 60.0

    # prepare output filename formatting
    fileprefix = Plate.filename.with_suffix("").name
    filesuffix = ("TPFratio", "SHGratio")[y_data == "SHGratio"]

    # the filename format (prefix_suffix_modality.pdf)
    filenamefmt = "{:s}_{:s}_{{:s}}.pdf".format(fileprefix, filesuffix)
    # the full target file includes the folder path
    targetfilefmt = str(filefolder / filenamefmt)

    dt = Plate.dt_postinject / 60.0
    t_array = np.arange(Plate.data.Nreads) * dt

    if global_scale:
        if y_data == "SHGratio":
            alldata = Plate.data.pSHG / Plate.data.sSHG
            global_min, global_max = (
                alldata.min(),
                alldata[np.isfinite(alldata)].max(),
            )

        elif y_data == "TPFratio":
            alldata = Plate.data.pTPF / Plate.data.sTPF
            global_min, global_max = (
                alldata.min(),
                alldata[np.isfinite(alldata)].max(),
            )

    with MultipagePDF(targetfilefmt.format("ptype"), nrows=3, ncols=4) as mpdf:

        for data, (conjugate, compound) in Plate.unique_experiments():
            # loop through each conjugate concentration data, conj_
            for uniqueconj, exptdata in data.items():

                conc_vec = exptdata["pTPF"]["concentrations"]

                if y_data == "TPFratio":
                    data_ = exptdata["pTPF"]["data"] / exptdata["sTPF"]["data"]
                    y_title = "TPF ratio (P/S)"

                if y_data == "SHGratio":
                    data_ = exptdata["pSHG"]["data"] / exptdata["sSHG"]["data"]
                    y_title = "SHG ratio (P/S)"

                Nreads, Nrep, Nconcs = data_.shape

                ax = mpdf.new_axis()
                ax.set_title("{:s}\n{:s}".format(uniqueconj, compound))

                ax.set_ylabel(y_title)
                ax.set_xlabel("Time, min")

                # now for each unique compound concentrations
                for c in range(Nconcs):
                    y_data_ = data_[:, :, c].mean(axis=1)
                    y_std_ = data_[:, :, c].std(axis=1)

                    ax.errorbar(
                        t_array, y_data_, yerr=y_std_, fmt="o-", markersize=5
                    )

                    if global_scale:
                        ax.set_ylim([global_min, global_max])

                    if ylims is not None:
                        ax.axhline(
                            y=0.65,
                            xmin=0.0,
                            xmax=t_array.max(),
                            linestyle="--",
                            color="#808080",
                        )
                        ax.set_ylim(ylims)
