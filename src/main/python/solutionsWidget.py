from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QDesktopWidget,
    QSlider,
    QLabel,
)
from PyQt5.QtCore import Qt
from MplWidgets import matplotlibWidget


class SolutionsCheckWidget(QDialog):
    """ A 1-by-2 plotting widget """

    def __init__(self, parent=None):
        super(SolutionsCheckWidget, self).__init__(parent)

        self.layout = QHBoxLayout()
        self.figure1 = matplotlibWidget()
        self.figure2 = matplotlibWidget()

        self.layout.addWidget(self.figure1)
        self.layout.addWidget(self.figure2)

        self.setLayout(self.layout)

        self.setGeometry(300, 300, 750, 420)

        self.show()

    def draw(self):

        self.figure1.canvas.figure.tight_layout()
        self.figure2.canvas.figure.tight_layout()
        self.figure1.canvas.refresh()
        self.figure2.canvas.refresh()


class AngleVariationsWidget(QDialog):
    """ 1-by-2 plotting widget with slider bars """

    def __init__(self, parent=None):
        super(AngleVariationsWidget, self).__init__(parent)

        cos = QDesktopWidget().availableGeometry().center()
        w, h = 1000, 500
        self.setGeometry(cos.x() - w // 2, cos.y() - h // 2, w, h)

        self.layout = QHBoxLayout()
        self.figure1 = matplotlibWidget()
        self.figure2 = matplotlibWidget()
        self.dialoglayout = QHBoxLayout()

        self.Playout = QVBoxLayout()
        self.Slayout = QVBoxLayout()
        self.Pslider = QSlider(Qt.Horizontal)
        self.Sslider = QSlider(Qt.Horizontal)

        self.Playout.addWidget(self.figure1)
        self.Playout.addWidget(self.Pslider)
        self.Slayout.addWidget(self.figure2)
        self.Slayout.addWidget(self.Sslider)

        self.dialoglayout.addLayout(self.Playout)
        self.dialoglayout.addLayout(self.Slayout)
        self.setLayout(self.dialoglayout)

        self.show()

    def draw(self):

        self.figure1.canvas.figure.tight_layout()
        self.figure2.canvas.figure.tight_layout()
        self.figure1.canvas.refresh()
        self.figure2.canvas.refresh()


if __name__ == "__main__":

    from PyQt5.QtWidgets import QApplication
    import sys

    # sys.path.append("C:\\Users\\delna\\Apps\\easyAMPS\\src\\main\\python")
    print("Running as main.... ")
    # to test the dialog box by via executing as python script
    app = QApplication(sys.argv)

    print("Instantiating AngleVariationsWidget()")
    form = AngleVariationsWidget()
    form.show()

    print("Event loop started.")
    sys.exit(app.exec_())
