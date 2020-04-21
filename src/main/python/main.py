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
import matplotlib.pyplot as plt
from AMPS.solvers import compute_angles
from AMPS.viz import visualize_AMPS_solution
from solutionsWidget import SolutionsCheckWidget
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


class AppContext(ApplicationContext):
    def run(self):
        window = MainWindow()
        window.show()
        return self.app.exec_()


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setupUi(self)
        self.setWindowTitle("easyAMPS v0.1 (Biodesy, Inc.)")

        # create a toolbar
        toolbar = QToolBar("AMPS toolbar")
        self.addToolBar(toolbar)

        fit_phases_button = QAction("Fit phases", self)
        fit_phases_button.setStatusTip("Determine phase difference from loaded data")
        fit_phases_button.triggered.connect(self.determine_phases)
        toolbar.addAction(fit_phases_button)

        plot_fits_button = QAction("Plot fits", self)
        plot_fits_button.setStatusTip("Plots the result of phase determination")
        plot_fits_button.triggered.connect(self.plot_phase_determination)
        toolbar.addAction(plot_fits_button)

        # add some empty dataframe to the angle calculator table
        calcanglesdf = pd.DataFrame(
            {"SHGratio": [""], "TPFratio": [""], "angle": [""], "distribution": [""],}
        )
        self.calculatorTableWidget.setDataFrame(calcanglesdf)

        # setup connections
        self.actionOpen.triggered.connect(self.openfile)
        self.actionExit.triggered.connect(sys.exit)
        self.computeAnglesButton.clicked.connect(self.computeAngles)
        self.checkSolutionsButton.clicked.connect(self.checkSolutions)

    def openfile(self):
        """ File > Open dialog box """

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, fileType = QFileDialog.getOpenFileName(
            self, "Open AMPS raw data", "", "CSV file (*.csv)", options=options
        )

        if fileName:
            self.rawdataframe = pd.read_csv(fileName)
            self.tableWidget.setDataFrame(self.rawdataframe)

        else:
            return False

    def determine_phases(self):
        pass

    def plot_phase_determination(self):
        pass

    def computeAngles(self):
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

    def checkSolutions(self):

        # get selected points
        selectedsolutions = self.calculatorTableWidget.selectedIndexes()
        selectedrows = list(set([sel.row() for sel in selectedsolutions]))

        shgratios = []
        tpfratios = []

        for row in selectedrows:
            _shgratio = self.calculatorTableWidget.df.loc[row, "SHGratio"]
            _tpfratio = self.calculatorTableWidget.df.loc[row, "TPFratio"]
            if _shgratio is not "":
                shgratios.append(toNumber(_shgratio))
            if _tpfratio is not "":
                tpfratios.append(toNumber(_tpfratio))

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
