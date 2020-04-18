from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5 import QtCore
from fbs_runtime.application_context.PyQt5 import ApplicationContext
from pathlib import Path
import sys
import os
import pandas as pd

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

root = Path(__file__).parent

Ui_MainWindow, QtBaseClass = uic.loadUiType(str(root / "easyAMPS_maingui.ui"))

class AppContext(ApplicationContext):
    def run(self):
        window = MainWindow()
        window.show()
        return self.app.exec_()


class pandasModel(QtCore.QAbstractTableModel):
    def __init__(self, dataframe):
        QtCore.QAbstractTableModel.__init__(self)
        self._dataframe = dataframe

    def rowCount(self, parent=None):
        return self._dataframe.shape[0]

    def columnCount(self, parent=None):
        return self._dataframe.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._dataframe.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._dataframe.columns[col]
        return None


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.actionOpen.triggered.connect(self.openfile)
        self.actionExit.triggered.connect(sys.exit)

    def openfile(self):
        """ File > Open dialog box """

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, fileType = QFileDialog.getOpenFileName(
            self, "Open AMPS raw data", "", "CSV file (*.csv)", options=options
        )

        if fileName:
            self.rawdataframe = pd.read_csv(fileName)
            tablemodel = pandasModel(self.rawdataframe)
            self.tableView.setModel(tablemodel)
            self.tableView.resizeColumnsToContents()

        else:
            return False


if __name__ == "__main__":
    appctxt = AppContext()  # 1. Instantiate ApplicationContext
    exit_code = appctxt.run()
    sys.exit(exit_code)
