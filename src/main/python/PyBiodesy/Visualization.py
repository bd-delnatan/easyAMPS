"""

Visualization module for PyBiodesy

There are many cases where the majority of data processing is achieved by
standard features available in the Python science stack (e.g. NumPy, Pandas,
matplotlib, seaborn). However, one feature that I like is the idea of keeping
a unique datapoint, including all of its replicates as a single object, and
operations between objects should behave like a single number. The object is
implemented as a "Replicate" class in PyBiodesy.DataStructures

This causes some difficulty when trying to use such collection of objects
when plotting with the standard functions.

In addition, I wanted an "errorbar" plotting function that not only plot the
errorbars, but also super-imposes individual datapoints as scatter plots on
the same plot.

"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
from PyBiodesy.DataStructures import stretch_xy
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable


class MidpointNormalize(mpl.colors.Normalize):
    """
    to normalized min,middle,max of a colormap. Great for diverging colormaps

    Usage::

        plt.imshow(mat, norm=MidpointNormalize(.., .., ..,))

    """

    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def scatterplot(x, y, ax=None, fmt="o", **kwargs):
    """

    scatter plot given x,y where x is a vector and y is a list of 'Replicate'

    """

    if ax is None:
        fig, ax = plt.subplots()

    ymean = [f.mean for f in y]
    ystd = [f.std for f in y]

    lx, ly = stretch_xy(x, [f.values for f in y])

    if "label" in kwargs:
        legend_label = kwargs["label"]
        # remove key from dictionary
        del kwargs["label"]
    else:
        legend_label = None

    pts2d, = ax.plot(lx, ly, "o", alpha=0.4, mec="none", **kwargs)

    ax.errorbar(
        x,
        ymean,
        yerr=ystd,
        fmt=fmt,
        c=pts2d.get_color(),
        label=legend_label,
        **kwargs,
    )


def plot_traces(optres, figsize=(7, 4)):
    """ plots the time traces from an emcee run via lmfit """
    Npars = len(optres.params)
    chains = optres.params
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    parnames = optres.params.keys()

    for n in range(Npars):
        ax[n].plot(chains[:, :, n].T, "-", color="#808080", alpha=0.3)
        ax[n].set_title(parnames[n])


def visualize_plate_data(
    dataframe,
    datacolumn,
    title="Plate view",
    axis_handle=None,
    row_name="Read Row",
    col_name="Read Column",
    colorbar_orientation="vertical",
    colorbar=False,
    colorbar_kws=None,
    imshow_kws=None,
    **kwargs,
):
    """ do the augmented matplotlib.pyplot.imshow for 384-well plate

    The extra arguments are passed onto pyplot.imshow(). This should be
    re-factored as a class method because it could be used for any plate
    data. The only thing that needs to be handled is the experiment-specific
    plate boundaries (self.height and self.width parameters). You can use
    the vmin, vmax argument to explicitly set the color mapping. Extra
    keyworded arguments are passed to Axes.imshow().

    Args:
        dataframe(pandas DataFrame): input data frame of a 384-well plate.
            The input must contain a column with well coordinates (e.g. Row,
            and Column columns), and a data column.
        datacolumn(str): the name of column containing data to visualize.
        title(str): the title of the plot.
        axis_handle(matplotlib Axes, optional): an Axes object on which to
            render the plot. Default is None.
        row_name(str): column header name to be used as rows.
        col_name(str): column header name to be used as columns.
        colorbar_orientation(str): "horizontal" or "vertical".
        colorbar(bool): if True, colorbar will be shown.
        **kwargs: extra keyworded arguments to be passed to Axes.imshow().
    """

    # create an array from the data frame

    rowlabels = dataframe[row_name].unique().tolist()
    collabels = dataframe[col_name].unique().tolist()
    Nrows = len(rowlabels)
    Ncolumns = len(collabels)

    row_idx = {letter: number for number, letter in enumerate(rowlabels)}
    col_idx = {colnum: number for number, colnum in enumerate(collabels)}
    array = np.zeros((Nrows, Ncolumns))
    for rowindex, rowitem in dataframe.iterrows():
        i, j = row_idx[rowitem[row_name]], col_idx[rowitem[col_name]]
        array[i, j] = rowitem[datacolumn]

    if axis_handle is None:
        _view = plt.imshow
        _title = plt.title
        _hlines = plt.hlines
        _vlines = plt.vlines

        def _yticks(y, ylabels, fontdict):
            plt.yticks(y, ylabels, **fontdict)

        def _xticks(x, xlabels, fontdict):
            plt.xticks(x, xlabels, **fontdict)

    else:

        _view = axis_handle.imshow
        _title = axis_handle.set_title
        _hlines = axis_handle.hlines
        _vlines = axis_handle.vlines
        divider = make_axes_locatable(axis_handle)

        caxis_orientation = (
            "right" if colorbar_orientation == "vertical" else "top"
        )

        cax = divider.append_axes(caxis_orientation, size="5%", pad=0.05)

        def _xticks(x, xlabels, **kwargs):
            axis_handle.set_xticks(x)
            axis_handle.set_xticklabels(xlabels, **kwargs)

        def _yticks(y, ylabels, **kwargs):
            axis_handle.set_yticks(y)
            axis_handle.set_yticklabels(ylabels, **kwargs)

    if imshow_kws is not None:
        mappable = _view(array, **imshow_kws)
    else:
        mappable = _view(array)

    _title(title)
    _hlines(
        y=np.arange(1 + Nrows) - 0.5,
        xmin=-0.5,
        xmax=Ncolumns - 0.5,
        colors="#999999",
    )
    _vlines(
        x=np.arange(1 + Ncolumns) - 0.5,
        ymin=-0.5,
        ymax=Nrows - 0.5,
        colors="#999999",
    )
    _yticks(
        np.arange(len(rowlabels)), rowlabels, fontdict={"fontsize": "small"}
    )
    _xticks(
        np.arange(len(collabels)), collabels, fontdict={"fontsize": "small"}
    )

    if colorbar:
        if axis_handle is None:
            if colorbar_kws is not None:
                plt.colorbar(
                    orientation=colorbar_orientation, pad=0.15, **colorbar_kws
                )
            else:
                plt.colorbar(orientation=colorbar_orientation, pad=0.15)
        else:
            if colorbar_kws is not None:
                plt.colorbar(
                    mappable,
                    orientation=colorbar_orientation,
                    cax=cax,
                    **colorbar_kws,
                )
            else:
                plt.colorbar(
                    mappable, orientation=colorbar_orientation, cax=cax
                )


def sgolay2d(z, window_size, order, derivative=None):
    """
    """
    # number of terms in the polynomial expression
    n_terms = (order + 1) * (order + 2) / 2.0

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    if window_size ** 2 < n_terms:
        raise ValueError("order is too high for the window size")

    half_size = window_size // 2

    # exponents of the polynomial.
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [(k - n, n) for k in range(order + 1) for n in range(k + 1)]

    # coordinates of points
    ind = np.arange(-half_size, half_size + 1, dtype=np.float64)
    dx = np.repeat(ind, window_size)
    dy = np.tile(ind, [window_size, 1]).reshape(window_size ** 2)

    # build matrix of system of equation
    A = np.empty((window_size ** 2, len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (dx ** exp[0]) * (dy ** exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2 * half_size, z.shape[1] + 2 * half_size
    Z = np.zeros((new_shape))
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] = band - np.abs(
        np.flipud(z[1 : half_size + 1, :]) - band
    )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band + np.abs(
        np.flipud(z[-half_size - 1 : -1, :]) - band
    )
    # left band
    band = np.tile(z[:, 0].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs(
        np.fliplr(z[:, 1 : half_size + 1]) - band
    )
    # right band
    band = np.tile(z[:, -1].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, -half_size:] = band + np.abs(
        np.fliplr(z[:, -half_size - 1 : -1]) - band
    )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0, 0]
    Z[:half_size, :half_size] = band - np.abs(
        np.flipud(np.fliplr(z[1 : half_size + 1, 1 : half_size + 1])) - band
    )
    # bottom right corner
    band = z[-1, -1]
    Z[-half_size:, -half_size:] = band + np.abs(
        np.flipud(np.fliplr(z[-half_size - 1 : -1, -half_size - 1 : -1])) - band
    )

    # top right corner
    band = Z[half_size, -half_size:]
    Z[:half_size, -half_size:] = band - np.abs(
        np.flipud(Z[half_size + 1 : 2 * half_size + 1, -half_size:]) - band
    )
    # bottom left corner
    band = Z[-half_size:, half_size].reshape(-1, 1)
    Z[-half_size:, :half_size] = band - np.abs(
        np.fliplr(Z[-half_size:, half_size + 1 : 2 * half_size + 1]) - band
    )

    # solve system and convolve
    if derivative is None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, m, mode="valid")
    elif derivative == "col":
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -c, mode="valid")
    elif derivative == "row":
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode="valid")
    elif derivative == "both":
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return (
            scipy.signal.fftconvolve(Z, -r, mode="valid"),
            scipy.signal.fftconvolve(Z, -c, mode="valid"),
        )


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
