from matplotlib.backends.backend_qt5agg import (
    FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(111)
        super(MplCanvas, self).__init__(self.figure)

    def refresh(self):
        self.figure.canvas.draw()


class MplGridCanvas(FigureCanvas):
    def __init__(
        self, nrows=1, ncols=1, parent=None, width=5, height=4, dpi=100
    ):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.subplots(nrows=nrows, ncols=ncols)
        super(MplGridCanvas, self).__init__(self.figure)

    def refresh(self):
        self.figure.canvas.draw()


class matplotlibWidget(QWidget):
    def __init__(self, nrows=1, ncols=1, parent=None):
        QWidget.__init__(self, parent)
        self.canvas = MplCanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.vbl = QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)


class visAMPSWidget(QWidget):
    def __init__(self, nrows=1, ncols=1, parent=None):
        QWidget.__init__(self, parent)
        self.canvas = MplGridCanvas(nrows=2, ncols=2)
        self.lasttwinax = self.canvas.axes[1, 1].twinx()
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.vbl = QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)
