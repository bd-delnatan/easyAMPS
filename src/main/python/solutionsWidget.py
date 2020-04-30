from PyQt5.QtWidgets import QDialog, QHBoxLayout
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
