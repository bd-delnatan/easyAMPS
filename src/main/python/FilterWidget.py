from PyQt5.QtWidgets import (
    QApplication,
    QListWidget,
    QListWidgetItem,
    QDialog,
    QDesktopWidget,
    QHBoxLayout,
    QVBoxLayout,
    QComboBox,
    QLineEdit,
    QPushButton,
)
from PyQt5.QtCore import QAbstractListModel, Qt


class FilterDialog(QDialog):
    def __init__(self, column_names, parent=None):
        super(FilterDialog, self).__init__(parent)

        self.parent = parent

        cos = QDesktopWidget().availableGeometry().center()
        w = 500
        h = 400

        self.setGeometry(cos.x() - w // 2, cos.y() - h // 2, w, h)

        self.layout = QVBoxLayout()

        # filter items
        self.filter_layout = QHBoxLayout()
        self.column_names = QComboBox()
        for column in column_names:
            self.column_names.addItem(column)
        self.condition = QComboBox()
        self.condition.addItem("==")
        self.condition.addItem(">")
        self.condition.addItem("!=")
        self.value = QLineEdit()
        self.filter_layout.addWidget(self.column_names)
        self.filter_layout.addWidget(self.condition)
        self.filter_layout.addWidget(self.value)

        self.button_layout = QHBoxLayout()
        self.add_button = QPushButton("Add")
        self.remove_button = QPushButton("Remove")
        self.button_layout.addWidget(self.add_button)
        self.button_layout.addWidget(self.remove_button)

        # list widget
        self.filterlistwidget = QListWidget()

        self.layout.addLayout(self.filter_layout)
        self.layout.addLayout(self.button_layout)
        self.layout.addWidget(self.filterlistwidget)

        self.setLayout(self.layout)
        self.setWindowTitle("Filter")

        self.setup_button_connections()

        if parent is not None:
            existing_items = [f for f in self.parent._data_model.filters]
            for item in existing_items:
                self.filterlistwidget.insertItem(0, item)

        self.filterlistwidget.model().rowsInserted.connect(self.refresh_add)
        self.filterlistwidget.model().rowsRemoved.connect(self.refresh_remove)

    def refresh_add(self, index, start, end):
        Nitems = self.filterlistwidget.count()
        filterlist = [self.filterlistwidget.item(n).text() for n in range(Nitems)]

        for filt in filterlist:
            if filt not in self.parent._data_model.filters:
                self.parent._data_model.filters.append(filt)

        self.parent._data_model.filterData()

    def refresh_remove(self, index, start, end):
        Nitems = self.filterlistwidget.count()
        if Nitems > 0:
            filterlist = [self.filterlistwidget.item(n).text() for n in range(Nitems)]
        elif Nitems == 0:
            filterlist = []
        # remaining filters
        self.parent._data_model.filters = filterlist
        self.parent._data_model.filterData()

    def setup_button_connections(self):

        self.add_button.clicked.connect(self.add_filter)
        self.remove_button.clicked.connect(self.remove_filter)

    def add_filter(self):
        current_row = self.filterlistwidget.currentRow()
        # get filter information
        column_name = self.column_names.currentText()
        condition = self.condition.currentText()
        value = self.value.text()

        def isfloat(s):
            try:
                if (float(s) % 1.0) > 0:
                    return True
                else:
                    return False
            except:
                return False

        def isint(s):
            try:
                int(s)
                return True
            except:
                return False

        # compose filter entry
        if isfloat(value):
            expr = f"`{column_name:s}` {condition:s} {float(value):.4f}"
        elif isint(value):
            expr = f"`{column_name:s}` {condition:s} {int(value):d}"
        else:
            expr = f"`{column_name:s}` {condition:s} '{value:s}'"

        if not self.already_exists(expr):
            self.filterlistwidget.insertItem(current_row, expr)

    def remove_filter(self):
        current_row = self.filterlistwidget.currentRow()
        item = self.filterlistwidget.item(current_row)
        garbage = self.filterlistwidget.takeItem(current_row)
        del garbage

    def already_exists(self, item):
        Nitems = self.filterlistwidget.count()
        items = [self.filterlistwidget.item(n).text() for n in range(Nitems)]
        if item in items:
            return True
        else:
            return False


if __name__ == "__main__":

    import sys

    print("Running as main.... ")
    # to test the dialog box by via executing as python script
    app = QApplication(sys.argv)

    print("Instantiating filterDialog()")
    form = FilterDialog(["Green Eggs", "and", "ham"])
    form.show()
    print("Event loop started.")
    sys.exit(app.exec_())
