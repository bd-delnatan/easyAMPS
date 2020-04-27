from PyQt5 import uic
from PyQt5.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QAction,
    QStatusBar,
    QToolBar,
)
from fbs_runtime.application_context.PyQt5 import ApplicationContext
from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from AMPS.solvers import compute_angles
from AMPS import AMPSexperiment
from AMPS.viz import visualize_AMPS_solution, overview
from solutionsWidget import SolutionsCheckWidget
from CustomTable import alert
from AMPS import _AMPSboundaries

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

root = Path(__file__).parent

Ui_MainWindow, QtBaseClass = uic.loadUiType(str(root / "easyAMPS_maingui.ui"))

plt.style.use(str(root / "delnatan.mplstyle"))


def toNumber(x):

    try:
        # convert to float
        out = float(x)
    except ValueError:
        # if can't do it, return as string
        out = str(x)

    return out


def checkcolumns(columns):
    columnset = {"P-SHG", "S-SHG", "P-FLcorr", "S-FLcorr", "TPFratio", "frac_labeled"}
    inputset = set(columns)

    valid = columnset.issubset(inputset)

    if not valid:
        missing = list(columnset.intersection(inputset).symmetric_difference(columnset))
    else:
        missing = []

    return valid, missing


class AppContext(ApplicationContext):
    def run(self):
        window = MainWindow()
        window.show()
        return self.app.exec_()


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setupUi(self)
        self.maintitle = "easyAMPS v0.1 (Biodesy, Inc.)"
        self.currentfilepath = None

        self.setWindowTitle(self.maintitle)

        # create a toolbar
        toolbar = QToolBar("AMPS toolbar")
        self.addToolBar(toolbar)

        fit_phases_button = QAction("Plot 4ch data", self)
        fit_phases_button.setStatusTip("Plot 4-channel SHG/TPF data from Data Table")
        fit_phases_button.triggered.connect(self.visualize_4ch)
        toolbar.addAction(fit_phases_button)

        plot_fits_button = QAction("Fit phases", self)
        plot_fits_button.setStatusTip("Fit phase difference to 4-channel data")
        plot_fits_button.triggered.connect(self.fit_phase_difference)
        toolbar.addAction(plot_fits_button)

        # add some empty dataframe to the angle calculator table
        calcanglesdf = pd.DataFrame(
            {"TPFratio": [""], "SHGratio": [""], "angle": [""], "distribution": [""],}
        )
        self.calculatorTableWidget.setDataFrame(calcanglesdf)

        # setup connections
        self.actionOpen.triggered.connect(self.openfile)
        self.actionParse_raw_data.triggered.connect(self.open_parser_dialog)
        self.actionExit.triggered.connect(sys.exit)

        # button connections
        self.computeAnglesButton.clicked.connect(self.compute_angles)
        self.checkSolutionsButton.clicked.connect(self.check_solutions)
        self.correctSHGButton.clicked.connect(self.correct_SHG)

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
            self.tableWidget.setDataFrame(self.rawdataframe)
            newWindowTitle = f"{self.maintitle} : {fileName}"
            self.setWindowTitle(newWindowTitle)
        else:
            return False

    def visualize_4ch(self):
        # self.visualchecksWidget
        # get raw data
        data = self.tableWidget._data_model.df

        if data.empty:

            alert("Warning!", "No data is loaded")

            return False

        else:

            columnsOK, missing = checkcolumns(data.columns)

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

            self.visualchecksWidget.canvas.draw()

        else:
            warnstr = f"Columns : {missing} missing from data table."

            alert("Warning!", warnstr)

    def fit_phase_difference(self):

        data = self.tableWidget._data_model.df

        if self.currentfilepath is not None:

            exptname = self.currentfilepath.stem

        else:

            alert("Warning!", "No file has been loaded")
            return False

        if data.empty:

            alert("Warning!", "No data is loaded")
            return False

        else:

            columnsOK, missing = checkcolumns(data.columns)

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
            Pphase_sigma = self.experiment.Pphases.optres.params["delphi"].stderr
            Pphase_error = (
                np.rad2deg(Pphase_sigma) if Pphase_sigma is not None else np.nan
            )
            self.Pphase_sigma_label.setText(f"±{Pphase_error:.2f}")
            Sphase = self.experiment.sshg_phase
            Sphase_sigma = self.experiment.Sphases.optres.params["delphi"].stderr
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

            self.visualchecksWidget.canvas.draw()

    def correct_SHG(self):

        # get phase and background values from GUI
        phaseP = self.phasePspinBox.value()
        phaseS = self.phaseSspinBox.value()
        bgP = self.backgroundPspinBox.value()
        bgS = self.backgroundSspinBox.value()
        Pinflection = self.PinflectionSpinBox.value()
        Sinflection = self.SinflectionSpinBox.value()

        if Pinflection == 0.0:
            Pinflection = None
        if Sinflection == 0.0:
            Sinflection = None

        # create an experiment
        self.experiment = AMPSexperiment(self.tableWidget._data_model.df, name="wrk")
        self.experiment.apply_SHG_correction(
            force_phase_P=phaseP,
            force_phase_S=phaseS,
            force_PSHG_bg=bgP,
            force_SSHG_bg=bgS,
            P_inflection=Pinflection,
            S_inflection=Sinflection,
        )

        self.experiment.compute_SHGratio()
        self.tableWidget._data_model.df = self.experiment.data

    def compute_angles(self):
        # get data from table
        df = self.calculatorTableWidget.df.copy()
        df["SHGratio"] = df["SHGratio"].astype(float)
        df["TPFratio"] = df["TPFratio"].astype(float)
        sol = df.apply(compute_angles, axis=1)
        df[["angle", "distribution"]] = sol

        # reassign results back to widget
        self.calculatorTableWidget._data_model.df = df
        self.calculatorTableWidget.resizeColumnsToContents()

        self.anglecalc_mplwidget.canvas.axes.clear()
        self.anglecalc_mplwidget.canvas.axes.plot(
            sol["distribution"].values, sol["angle"].values, "o", c="darkred"
        )

        self.anglecalc_mplwidget.canvas.axes.set_xlabel("distribution")
        self.anglecalc_mplwidget.canvas.axes.set_ylabel("angle")
        self.anglecalc_mplwidget.canvas.axes.set_xlim([2.0, 70.0])
        self.anglecalc_mplwidget.canvas.axes.set_ylim([0.0, 89.9])
        self.anglecalc_mplwidget.canvas.draw()

    def open_parser_dialog(self):
        pass

    def check_solutions(self):

        # get selected points
        selectedsolutions = self.calculatorTableWidget.selectedIndexes()
        selectedrows = list(set([sel.row() for sel in selectedsolutions]))

        shgratios = []
        tpfratios = []

        for row in selectedrows:
            _shgratio = self.calculatorTableWidget._data_model.df.loc[row, "SHGratio"]
            _tpfratio = self.calculatorTableWidget._data_model.df.loc[row, "TPFratio"]
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


if __name__ == "__main__":
    appctxt = AppContext()  # 1. Instantiate ApplicationContext
    exit_code = appctxt.run()
    sys.exit(exit_code)
