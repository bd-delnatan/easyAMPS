"""

    Custom Table View

    04-20-2020:
        Copy & paste operations are supported
        Sorting by column is also supported


"""

from PyQt5.QtWidgets import (
    QTableView,
    QMenu,
    QInputDialog,
    QApplication,
    QMessageBox,
)
from PyQt5.QtCore import (
    Qt,
    QModelIndex,
    QAbstractTableModel,
    QVariant,
    pyqtSignal,
)
from PyQt5.QtGui import QKeySequence
from functools import partial
from io import StringIO
import pandas
from numpy import nan


def alert(title="Title", message="message"):
    msgbox = QMessageBox()
    msgbox.setText(message)
    msgbox.setWindowTitle(title)
    msgbox.exec_()


class DataFrameWidget(QTableView):

    cellClicked = pyqtSignal(int, int)

    def __init__(self, parent=None, df=None):
        """ DataFrame widget

        first argument needs to be parent because QtDesigner
        passes the parent QWidget when instantiating the object

        """
        super(DataFrameWidget, self).__init__(parent)

        self._data_model = DataFrameModel()
        self.setModel(self._data_model)

        if df is None:
            df = pandas.DataFrame()

        self._data_model.setDataFrame(df)

        # create header menu bindings
        self.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.horizontalHeader().customContextMenuRequested.connect(
            self._header_menu
        )

        self.verticalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.verticalHeader().customContextMenuRequested.connect(
            self._index_menu
        )

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._table_menu)

        # we intercept the clicked slot and connect to our own _on_click
        # to emit a custom signal
        self.clicked.connect(self._on_click)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Copy):
            self.copy()
        elif event.matches(QKeySequence.Paste):
            # load clipboard content as dataframe
            self.paste()
        else:
            # ignore the event, pass through normal event handler
            super(DataFrameWidget, self).keyPressEvent(event)

    def setDataFrame(self, dataframe):
        self._data_model.setDataFrame(dataframe)
        self.resizeColumnsToContents()

    def _on_click(self, index):
        if index.isValid():
            self.cellClicked.emit(index.row(), index.column())

    @property
    def df(self):
        """ returns the underlying data-frame """
        return self._data_model.df

    def copy(self):
        """ copy selected cells into clipboard """
        selindexes = self.selectedIndexes()

        if len(selindexes) < 1:
            # nothing is selected
            return None

        # create a DataFrame container to hold data
        container = pandas.DataFrame()

        for sel in selindexes:
            row = sel.row()
            col = sel.column()
            entry = sel.data()
            if entry:
                container.at[row, col] = entry

        str2copy = "\n".join(
            ["\t".join([f"{r}" for r in row]) for row in container.values]
        )

        QApplication.clipboard().setText(str2copy)

    def paste(self):
        """ paste data copied on clipboard """

        selindexes = self.selectedIndexes()

        if len(selindexes) != 1:
            alert(
                title="Alert",
                message="To paste into table, select a single cell",
            )
        else:
            clipboard = QApplication.clipboard().text()
            dfnew = pandas.read_csv(StringIO(clipboard), sep="\t", header=None)
            Nrows, Ncols = dfnew.shape
            # current position
            row_id, col_id = selindexes[0].row(), selindexes[0].column()
            # figure out if the current size of the table fits
            rowsize, colsize = self.df.shape

            Ncol_extra = (
                0 if col_id + Ncols <= colsize else col_id + Ncols - colsize
            )
            Nrow_extra = (
                0 if row_id + Nrows <= rowsize else row_id + Nrows - rowsize
            )

            if Ncol_extra > 0:
                self._data_model.insertColumns(colsize, count=Ncol_extra)
            if Nrow_extra > 0:
                self._data_model.insertRows(rowsize, count=Nrow_extra)

            self._data_model.layoutAboutToBeChanged.emit()
            self._data_model.df.iloc[
                row_id : row_id + Nrows, col_id : col_id + Ncols
            ] = dfnew.values
            self._data_model.layoutChanged.emit()

            self.resizeColumnsToContents()

    def _header_menu(self, pos):
        menu = QMenu(self)
        column_index = self.horizontalHeader().logicalIndexAt(pos)

        if column_index == -1:
            # out of bounds
            return

        menu.addAction(
            r"Rename column", partial(self.renameHeader, column_index)
        )
        menu.addSeparator()

        menu.addAction(
            r"Sort (Descending)",
            partial(
                self._data_model.sort, column_index, order=Qt.DescendingOrder
            ),
        )

        menu.addAction(
            r"Sort (Ascending)",
            partial(
                self._data_model.sort, column_index, order=Qt.AscendingOrder
            ),
        )

        menu.addSeparator()

        menu.addAction(
            r"Insert column <-",
            partial(self._data_model.insertColumns, column_index),
        )

        menu.addAction(
            r"Insert column ->",
            partial(self._data_model.insertColumns, column_index + 1),
        )

        menu.exec_(self.mapToGlobal(pos))

    def _index_menu(self, pos):
        menu = QMenu(self)
        row_index = self.verticalHeader().logicalIndexAt(pos)

        if row_index == -1:
            # out of bounds
            return

        menu.addAction(
            r"Insert row above", partial(self._data_model.insertRows, row_index)
        )

        menu.addAction(
            r"Insert row below",
            partial(self._data_model.insertRows, row_index + 1),
        )

        menu.addAction(r"Delete selected row(s)", self.removeSelectedRows)

        menu.exec_(self.mapToGlobal(pos))

    def _table_menu(self, pos):
        menu = QMenu(self)
        menu.addAction(r"Copy selected", self.copy)
        menu.exec_(self.mapToGlobal(pos))

    def renameHeader(self, column_index):

        newname, ok = QInputDialog.getText(
            self, "Rename column header", "New column name:"
        )
        newcolumns = self.df.columns.tolist()
        newcolumns[column_index] = newname
        self.df.columns = newcolumns

    def removeSelectedRows(self):
        selected_rows = sorted(
            list(set([sel.row() for sel in self.selectedIndexes()]))
        )

        while len(selected_rows) > 0:
            target = selected_rows.pop()
            self._data_model.removeRows(target)


class DataFrameModel(QAbstractTableModel):
    def __init__(self):
        super(DataFrameModel, self).__init__()
        self._df = pandas.DataFrame()
        self.Ncoladded = 0

    def setDataFrame(self, dataframe):
        # this uses the .setter method
        self.df = dataframe

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, dataframe):
        self.modelAboutToBeReset.emit()
        self._df = dataframe
        self.modelReset.emit()

    # table display functions
    def data(self, index, role):

        if role == Qt.DisplayRole:
            if not index.isValid():
                return QVariant()

            # for DisplayRole, return a string. For numbers, the library
            # automatically does this.
            value = self.df.iloc[index.row(), index.column()]

            if pandas.isnull(value):
                return QVariant()

            # you can format displayed strings here
            if isinstance(value, float):
                return f"{value:.5f}"

            if isinstance(value, int):
                return f"{value:d}"

            if isinstance(value, str):
                return f"{value:s}"

            return f"{value}"

        else:

            return None

    def headerData(self, pos, orientation, role):
        # early return if not for display
        if role != Qt.DisplayRole:
            return None

        # Horizontal (column) headers
        if orientation == Qt.Horizontal:
            return self.df.columns.tolist()[pos]

        # Vertical (index) headers
        if orientation == Qt.Vertical:
            return self.df.index.tolist()[pos]

    # end table display functions

    def rowCount(self, index):
        return self.df.shape[0]

    def columnCount(self, index):
        return self.df.shape[1]

    def flags(self, index):
        fl = super(DataFrameModel, self).flags(index)
        fl |= Qt.ItemIsEditable
        fl |= Qt.ItemIsSelectable
        fl |= Qt.ItemIsEnabled
        return fl

    # for data entry into cell
    def setData(self, index, value, role=Qt.EditRole):
        if index.isValid():
            row = index.row()
            column = index.column()

            if value == "":
                return False

            self._df.iloc[row, column] = value

            self.dataChanged.emit(index, index, (Qt.DisplayRole,))

            return True

        else:

            return False

    def sort(self, column_index, order=Qt.AscendingOrder):
        if column_index >= self.df.shape[1]:
            # out of bounds
            return None
        self.layoutAboutToBeChanged.emit()

        ascending = True if order == Qt.AscendingOrder else False

        self.df = self.df.sort_values(
            self.df.columns[column_index], ascending=ascending
        )
        self.layoutChanged.emit()

    def insertRows(self, position, count=1, index=QModelIndex()):

        self.beginInsertRows(QModelIndex(), position, position + count - 1)

        newrows = pandas.DataFrame(
            [[nan for i in range(self.df.shape[1])] for j in range(count)],
            columns=self.df.columns,
        )

        self.df = pandas.concat(
            [self.df.iloc[0:position], newrows, self.df.iloc[position:]],
            axis=0,
            ignore_index=True,
        )
        self.indexlabels = self.df.index

        self.endInsertRows()

        return True

    def removeRows(self, position, count=1, index=QModelIndex()):

        self.beginRemoveRows(QModelIndex(), position, position + count - 1)
        self.df = self.df.drop(self.df.index[position], axis=0)
        self.df.index = [i for i in range(self.df.shape[0])]
        self.endRemoveRows()

        return True

    def insertColumns(self, position, count=1, index=QModelIndex()):
        # to keep track of naming
        self.Ncoladded += 1
        self.beginInsertColumns(QModelIndex(), position, position + count - 1)
        # insert is an in-place operation
        self.df.insert(position, f"NewColumn{self.Ncoladded}", value=nan)
        self.endInsertColumns()

        return True

    def removeColumns(self, position, count=1, index=QModelIndex()):
        self.beginRemoveColumns(QModelIndex(), position, position + count - 1)
        self.df = self.df.drop(self.headers[position], axis=1)
        self.endRemoveColumns()

        return True