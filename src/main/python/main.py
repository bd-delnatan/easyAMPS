from PyQt5.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QAction,
    QToolBar,
)
from fbs_runtime.application_context.PyQt5 import ApplicationContext
from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
from AMPS.solvers import compute_angles, solve_AMPS
from AMPS import AMPSexperiment, return_signs
from AMPS.functions import nP_SHG, nS_SHG, nP_TPF, nS_TPF
from AMPS.viz import visualize_AMPS_solution, overview
from PyBiodesy.DataStructures import correct_SHG
from solutionsWidget import SolutionsCheckWidget
from scriptWriter import ScriptWriterDialog
from CustomTable import alert
from AMPS import _AMPSboundaries
from easyAMPS_maingui import Ui_MainWindow
from DebugWindow import ScriptWindow

__VERSION__ = "0.1.4"


# needed to properly scale high DPI screens in Windows OS
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

# Ui_MainWindow, QtBaseClass = uic.loadUiType(str(root / "easyAMPS_maingui.ui"))
mpl.rc(
    "font", **{"size": 10, "family": "sans-serif"},
)
mpl.pyplot.style.use("bmh")


def toNumber(x):

    try:
        # convert to float
        out = float(x)
    except ValueError:
        # if can't do it, return as string
        out = str(x)

    return out


def checkcolumns(columns):
    columnset = {
        "P-SHG",
        "S-SHG",
        "P-FLcorr",
        "S-FLcorr",
        "TPFratio",
        "frac_labeled",
    }
    inputset = set(columns)

    valid = columnset.issubset(inputset)

    if not valid:
        missing = list(
            columnset.intersection(inputset).symmetric_difference(columnset)
        )
    else:
        missing = []

    return valid, missing


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        voffset = 3 if height > 0 else -15
        ax.annotate(
            "{:.1f}%".format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, voffset),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


class AppContext(ApplicationContext):
    def run(self):
        window = MainWindow()
        window.show()
        return self.app.exec_()


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.maintitle = f"easyAMPS v{__VERSION__} (Biodesy, Inc.)"
        self.currentfilepath = None

        # initialize njit (throwaway solution)
        _ = solve_AMPS(20, 1.3)

        # To make Mac/Windows more coherent disable Native Mac menubar
        self.menuBar().setNativeMenuBar(False)

        self.setWindowTitle(self.maintitle)

        # create a toolbar
        toolbar = QToolBar("AMPS toolbar")
        self.addToolBar(toolbar)

        fit_phases_button = QAction("Plot 4ch data", self)
        fit_phases_button.setStatusTip(
            "Plot 4-channel SHG/TPF data from Data Table"
        )
        fit_phases_button.triggered.connect(self.visualize_4ch)
        toolbar.addAction(fit_phases_button)

        plot_fits_button = QAction("Fit phases", self)
        plot_fits_button.setStatusTip("Fit phase difference to 4-channel data")
        plot_fits_button.triggered.connect(self.fit_phase_difference)
        toolbar.addAction(plot_fits_button)

        toolbar.addSeparator()
        debug_button = QAction("Open debugger", self)
        debug_button.triggered.connect(self.opendebugger)
        toolbar.addAction(debug_button)

        # add some empty dataframe to the angle calculator table
        calcanglesdf = pd.DataFrame(
            {
                "TPFratio": [""],
                "SHGratio": [""],
                "angle": [""],
                "distribution": [""],
                "label": [""],
            }
        )
        self.calculatorTableWidget.setDataFrame(calcanglesdf)

        # initialize empty dataframe with the minimum column headers
        emptydataframe = pd.DataFrame(
            {
                "P-SHG": [""],
                "S-SHG": [""],
                "P-FLcorr": [""],
                "S-FLcorr": [""],
                "frac_labeled": [""],
                "TPFratio": [""],
            }
        )
        self.tableWidget.setDataFrame(emptydataframe)

        # setup connections
        self.actionOpen.triggered.connect(self.openfile)
        self.actionSaveCSV.triggered.connect(self.savefile)
        self.actionAMPS_script_editor.triggered.connect(
            self.open_AMPS_script_editor
        )
        self.actionExit.triggered.connect(sys.exit)

        # button connections
        self.computeAnglesButton.clicked.connect(self.compute_angles)
        self.checkSolutionsButton.clicked.connect(self.check_solutions)
        self.correctSHGButton.clicked.connect(self.correct_SHG)
        self.predictSignalButton.clicked.connect(self.predict_signals)

    def opendebugger(self):
        dlg = ScriptWindow(parent=self)
        dlg.exec_()

    def openfile(self):
        """ File > Open dialog box """

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, fileType = QFileDialog.getOpenFileName(
            self, "Open AMPS raw data", "", "CSV file (*.csv)", options=options
        )

        if fileName:
            self.currentfilepath = Path(fileName)
            self.rawdataframe = pd.read_csv(fileName)
            columns = self.rawdataframe.columns
            replacements = {
                c: c.replace("#", "Num") for c in columns if "#" in c
            }
            self.rawdataframe.rename(columns=replacements, inplace=True)

            self.tableWidget.setDataFrame(self.rawdataframe)
            newWindowTitle = f"{self.maintitle} : {fileName}"
            self.setWindowTitle(newWindowTitle)
        else:
            return False

    def savefile(self):

        data = self.tableWidget._data_model.df

        if data.empty:
            alert("Warning", "Data table is empty")
        else:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            fileName, fileType = QFileDialog.getSaveFileName(
                self,
                "Save data table",
                "",
                "CSV files (*.csv);;Excel file (*.xlsx)",
                options=options,
            )

            extension = ".csv" if "CSV" in fileType else ".xlsx"
            targetfile = Path(fileName).with_suffix(extension)

            if extension == ".csv":
                data.to_csv(targetfile, index=False)
            elif extension == ".xlsx":
                data.to_excel(targetfile, index=False)

    def open_AMPS_script_editor(self):

        scripteditor_dialog = ScriptWriterDialog()
        scripteditor_dialog.exec_()

    def visualize_4ch(self):
        # self.visualchecksWidget
        # get raw data
        data = self.tableWidget.getVisibleData()

        if data.empty:
            alert("Warning!", "No data is loaded")
            return False
        else:
            columnsOK, missing = checkcolumns(data.columns)
            fraclabeled = data["frac_labeled"].unique().tolist()

            if not columnsOK:
                alert("Warning!", f"Columns {missing} are missing")
                return False

            if len(fraclabeled) < 3:
                alert("Warning!", f"You need at least 2 frac_labeled")
                return False

        # do checks on columns
        if columnsOK:

            # create a shorthand for the internal axes
            ax = self.visualchecksWidget.canvas.axes

            # last axis for linearity check
            self.visualchecksWidget.lasttwinax.clear()

            # clear previous plots (if any)

            for axes in ax.flat:
                axes.cla()

            overview(
                data,
                fighandles=(self.visualchecksWidget.canvas.figure, ax),
                twin_ax=self.visualchecksWidget.lasttwinax,
            )

            self.visualchecksWidget.canvas.refresh()

        else:
            warnstr = f"Columns : {missing} missing from data table."

            alert("Warning!", warnstr)

    def fit_phase_difference(self):

        data = self.tableWidget.getVisibleData()

        if self.currentfilepath is not None:
            exptname = self.currentfilepath.stem
        else:
            exptname = "DataNotLoaded"

        if data.empty:
            alert("Warning!", "No data is loaded")
            return False
        else:
            columnsOK, missing = checkcolumns(data.columns)
            fraclabeled = data["frac_labeled"].unique().tolist()

            if not columnsOK:
                alert("Warning!", f"Columns {missing} are missing")
                return False

            if len(fraclabeled) < 3:
                alert("Warning!", f"You need at least 2 frac_labeled")
                return False

        if columnsOK:
            self.experiment = AMPSexperiment(data, name=exptname)
            self.experiment.fit_phases()

            fitmsg = "P-polarization:\n"
            fitmsg += self.experiment.Pphases.optres.message
            fitmsg += "\nS-Polarization:\n"
            fitmsg += self.experiment.Sphases.optres.message

            alert("Phase determination", fitmsg)

            # extract parameters and put into GUI
            Pphase = self.experiment.pshg_phase
            Pphase_sigma = self.experiment.Pphases.optres.params[
                "delphi"
            ].stderr
            Pphase_error = (
                np.rad2deg(Pphase_sigma) if Pphase_sigma is not None else np.nan
            )
            self.Pphase_sigma_label.setText(f"±{Pphase_error:.2f}")
            Sphase = self.experiment.sshg_phase
            Sphase_sigma = self.experiment.Sphases.optres.params[
                "delphi"
            ].stderr
            Sphase_error = (
                np.rad2deg(Sphase_sigma) if Sphase_sigma is not None else np.nan
            )
            self.Sphase_sigma_label.setText(f"±{Sphase_error:.2f}")

            if Pphase > np.pi / 2.0:
                # destructive interference
                self.PinflectionSpinBox.setValue(self.experiment.p_inflection)
            else:
                # constructive has no inflection point
                self.PinflectionSpinBox.setValue(0.0)

            if Sphase > np.pi / 2.0:
                # destructive intereference
                self.SinflectionSpinBox.setValue(self.experiment.s_inflection)
            else:
                self.SinflectionSpinBox.setValue(0.0)

            Pbg = self.experiment.pshg_bg
            Sbg = self.experiment.sshg_bg

            self.phasePspinBox.setValue(np.rad2deg(Pphase))
            self.backgroundPspinBox.setValue(Pbg)
            self.phaseSspinBox.setValue(np.rad2deg(Sphase))
            self.backgroundSspinBox.setValue(Sbg)

            # create a shorthand for the internal axes
            ax = self.visualchecksWidget.canvas.axes

            # last axis for linearity check
            self.visualchecksWidget.lasttwinax.clear()

            # clear previous plots (if any)

            for axes in ax.flat:
                axes.cla()

            overview(
                data,
                fighandles=(self.visualchecksWidget.canvas.figure, ax),
                twin_ax=self.visualchecksWidget.lasttwinax,
                experiment=self.experiment,
            )

            self.visualchecksWidget.canvas.refresh()

    def correct_SHG(self):

        # get phase and background values from GUI
        # convert to radians
        phaseP = self.phasePspinBox.value() * np.pi / 180.0
        phaseS = self.phaseSspinBox.value() * np.pi / 180.0
        bgP = self.backgroundPspinBox.value()
        bgS = self.backgroundSspinBox.value()
        Pinflection = self.PinflectionSpinBox.value()
        Sinflection = self.SinflectionSpinBox.value()

        if Pinflection == 0.0:
            Pinflection = None
        if Sinflection == 0.0:
            Sinflection = None

        df = self.tableWidget._data_model.df.apply(
            pd.to_numeric, errors="ignore"
        )

        p_signs = return_signs(df["P-FLcorr"].values, Pinflection)
        s_signs = return_signs(df["S-FLcorr"].values, Sinflection)

        correct_SHG(df, bgP, bgS, phaseP, phaseS, p_signs, s_signs)
        df["SHGratio"] = df["P-SHGcorr"] / df["S-SHGcorr"]

        self.tableWidget._data_model.df = df.copy()

    def compute_angles(self):
        # get data from table
        df = self.calculatorTableWidget._data_model.df.copy()
        df["SHGratio"] = df["SHGratio"].astype(float)
        df["TPFratio"] = df["TPFratio"].astype(float)
        sol = df.apply(compute_angles, axis=1)
        df[["angle", "distribution"]] = sol

        df.loc[(df["label"] == "") | (df["label"].isna()), "label"] = "N/A"

        # reassign results back to widget
        self.calculatorTableWidget._data_model.df = df
        self.calculatorTableWidget.resizeColumnsToContents()

        self.anglecalc_mplwidget.canvas.axes.clear()

        unique_labels = df["label"].unique().tolist()

        if len(unique_labels) > 1:
            for label in unique_labels:
                wrk = df[df["label"] == label]
                self.anglecalc_mplwidget.canvas.axes.plot(
                    wrk["distribution"].values,
                    wrk["angle"].values,
                    "o",
                    label=label,
                )
        else:
            label = df["label"].unique().tolist()[0]
            self.anglecalc_mplwidget.canvas.axes.plot(
                df["distribution"].values, df["angle"].values, "o", label=label,
            )

        self.anglecalc_mplwidget.canvas.axes.legend(
            bbox_to_anchor=(1.10, 0.95), fontsize=8.5
        )

        self.anglecalc_mplwidget.canvas.axes.set_xlabel("distribution")
        self.anglecalc_mplwidget.canvas.axes.set_ylabel("angle")
        self.anglecalc_mplwidget.canvas.axes.set_xlim([2.0, 70.0])
        self.anglecalc_mplwidget.canvas.axes.set_ylim([0.0, 89.9])
        self.anglecalc_mplwidget.canvas.refresh()

    def check_solutions(self):

        # get selected points
        selectedsolutions = self.calculatorTableWidget.selectedIndexes()
        selectedrows = list(set([sel.row() for sel in selectedsolutions]))

        shgratios = []
        tpfratios = []

        for row in selectedrows:
            _shgratio = self.calculatorTableWidget._data_model.df.loc[
                row, "SHGratio"
            ]
            _tpfratio = self.calculatorTableWidget._data_model.df.loc[
                row, "TPFratio"
            ]
            if _shgratio != "":
                shgratios.append(toNumber(_shgratio))
            else:
                shgratios.append(np.nan)
            if _tpfratio != "":
                tpfratios.append(toNumber(_tpfratio))
            else:
                tpfratios.append(np.nan)

        dlg1 = SolutionsCheckWidget()
        dlg1.setWindowTitle("AMPS solution visual checks")

        dlg1.figure2.canvas.axes.fill_between(
            _AMPSboundaries.x,
            _AMPSboundaries.y_lower,
            _AMPSboundaries.y_upper,
            ec="darkgreen",
            linewidth=1.5,
            fc="mediumseagreen",
            alpha=0.4,
        )
        dlg1.figure2.canvas.axes.set_xlabel("TPF ratio")
        dlg1.figure2.canvas.axes.set_ylabel("SHG ratio")
        dlg1.figure2.canvas.axes.set_ylim([0, 40])
        dlg1.figure2.canvas.axes.set_xlim([0, 7])

        for _tpf, _shg in zip(tpfratios, shgratios):
            visualize_AMPS_solution(_shg, _tpf, ax=dlg1.figure1.canvas.axes)
            dlg1.figure2.canvas.axes.plot(_tpf, _shg, "o", c="rebeccapurple")

        # draw both plots
        dlg1.draw()
        # execute dialog box
        dlg1.exec_()

    def predict_signals(self):
        def _compute_signals(μ, σ):
            μ_rad, σ_rad = np.deg2rad(μ), np.deg2rad(σ)
            return np.array(
                [
                    nP_SHG(μ_rad, σ_rad),
                    nS_SHG(μ_rad, σ_rad),
                    nP_TPF(μ_rad, σ_rad),
                    nS_TPF(μ_rad, σ_rad),
                ]
            )

        ref_angle = self.referenceTiltSpinBox.value()
        ref_dist = self.referenceDistributionSpinBox.value()
        arg_angle = self.targetTiltSpinBox.value()
        arg_dist = self.targetDistributionSpinBox.value()

        if (ref_angle, ref_dist) == (arg_angle, arg_dist):
            alert("Warning", "Reference and final values are the same")
        else:
            ref_intensities = _compute_signals(ref_angle, ref_dist)
            arg_intensities = _compute_signals(arg_angle, arg_dist)

            # compute relative signal changes as a percentage
            signal_changes = 100.0 * (
                (arg_intensities - ref_intensities) / ref_intensities
            )

            bar_labels = ["P-SHG", "S-SHG", "P-FL", "S-FL"]
            xpos = np.arange(1, 5)

            # clear the axis
            self.predictedSignalWidget.canvas.axes.clear()

            # do the actual plotting
            bars1 = self.predictedSignalWidget.canvas.axes.bar(
                xpos, signal_changes
            )

            autolabel(bars1, self.predictedSignalWidget.canvas.axes)

            self.predictedSignalWidget.canvas.axes.grid(b=False, axis="x")

            self.predictedSignalWidget.canvas.axes.set_xticks(xpos)
            self.predictedSignalWidget.canvas.axes.set_xticklabels(bar_labels)
            self.predictedSignalWidget.canvas.axes.set_xlabel("Channels")
            self.predictedSignalWidget.canvas.axes.set_ylabel(
                r"% signal change"
            )
            self.predictedSignalWidget.canvas.refresh()


if __name__ == "__main__":
    appctxt = AppContext()  # 1. Instantiate ApplicationContext
    exit_code = appctxt.run()
    sys.exit(exit_code)
