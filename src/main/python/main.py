from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from fbs_runtime.application_context.PyQt5 import ApplicationContext
from pathlib import Path
import sys
import pandas as pd

root = Path(__file__).parent


class AppContext(ApplicationContext):
    def run(self):
        window = MainWindow()
        window.show()
        return self.app.exec_()


class MainWindow(QMainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi(str(root / "easyAMPS_maingui.ui"), self)

        self.actionOpen.triggered.connect(self.openfile)

    def openfile(self):

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, fileType = QFileDialog.getOpenFileName(
            self, "Open AMPS raw data", "", "CSV file (*.csv)", options=options
        )

        if fileName:
            self.rawdataframe = pd.read_csv(fileName)
        else:
            return False


if __name__ == "__main__":
    appctxt = AppContext()  # 1. Instantiate ApplicationContext
    exit_code = appctxt.run()
    sys.exit(exit_code)
