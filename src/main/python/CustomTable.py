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
from PyQt5.QtGui import QKeySequence, QBrush
from functools import partial
from io import StringIO
from FilterWidget import FilterDialog
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

        self.setDataFrame(df)

        # create (horizontal/top) header menu bindings
        self.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.horizontalHeader().customContextMenuRequested.connect(
            self._header_menu
        )

        # create (vertical/side/row) header menu bindings
        self.verticalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.verticalHeader().customContextMenuRequested.connect(
            self._index_menu
        )

        # create custom QTableView menu bindings
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._table_menu)

        # we intercept the clicked slot and connect to our own _on_click
        # to emit a custom signal
        self.clicked.connect(self._on_click)

    def keyPressEvent(self, event):
        # intercept "Copy" and "Paste" key combinations or delete
        if event.matches(QKeySequence.Copy):
            self.copy()
        elif event.matches(QKeySequence.Paste):
            # load clipboard content as dataframe
            self.paste()
        elif event.matches(QKeySequence.Delete):
            self.clear()
        else:
            # ignore the event, pass through normal event handler
            super(DataFrameWidget, self).keyPressEvent(event)

    def setDataFrame(self, dataframe):
        self._data_model.resetFilter()
        self._data_model.resetExcluded()
        self._data_model.setDataFrame(dataframe)
        self.resizeColumnsToContents()

    def _on_click(self, index):
        if index.isValid():
            self.cellClicked.emit(index.row(), index.column())

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
            else:
                # if cell empty, fill with nan
                container.at[row, col] = nan

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

            if clipboard is None or clipboard == "":
                return

            dfnew = pandas.read_csv(StringIO(clipboard), sep="\t", header=None)
            Nrows, Ncols = dfnew.shape
            # current position
            row_id, col_id = selindexes[0].row(), selindexes[0].column()
            # figure out if the current size of the table fits
            rowsize, colsize = self._data_model.df.shape

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

    def clear(self):
        """ clear selected cell contents """
        selindexes = self.selectedIndexes()

        self._data_model.layoutAboutToBeChanged.emit()

        for sel in selindexes:
            row = sel.row()
            col = sel.column()

            self._data_model.df.iloc[row, col] = nan

        self._data_model.layoutChanged.emit()

    def _header_menu(self, pos):
        """ context menu for header cells """
        menu = QMenu(self)
        column_index = self.horizontalHeader().logicalIndexAt(pos)

        if column_index == -1:
            # out of bounds
            return

        menu.addAction(
            r"Rename column", partial(self.renameHeader, column_index)
        )
        menu.addSeparator()

        menu.addAction(r"Copy selected header", self.copyHeader)
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

        menu.addAction(r"Delete selected column(s)", self.removeSelectedColumns)

        menu.exec_(self.mapToGlobal(pos))

    def _index_menu(self, pos):
        """ context menu for index/row cells """
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
        menu.addAction(r"Paste", self.paste)
        menu.addAction(r"Clear contents", self.clear)
        menu.addSeparator()
        menu.addAction(r"Exclude selected row(s)", self.exclude_selected_rows)
        menu.addSeparator()
        menu.addAction(r"Remove exclusion", self.remove_exlusion)
        menu.addSeparator()
        menu.addAction(r"Set filter", self.prototype_set_filter)
        menu.exec_(self.mapToGlobal(pos))

    def copyHeader(self):

        selindexes = self.selectedIndexes()
        selcolumns = sorted(
            list(set([sel.column() for sel in self.selectedIndexes()]))
        )

        if len(selindexes) < 1:
            # nothing is selected
            return None

        # create a DataFrame container to hold data
        headernames = []

        for col in selcolumns:
            headernames.append(self._data_model.df.columns[col])

        str2copy = "\t".join(headernames)
        QApplication.clipboard().setText(str2copy)

    def renameHeader(self, column_index):

        newname, ok = QInputDialog.getText(
            self, "Rename column header", "New column name:"
        )

        newname = newname.strip()

        # name can't be empty
        if newname != "":
            newcolumns = self._data_model.df.columns.tolist()
            newcolumns[column_index] = newname
            self._data_model.df.columns = newcolumns

    def removeSelectedRows(self):
        # get the number of rows
        Nrows = self._data_model.df.shape[0]

        if Nrows > 1:
            selected_rows = sorted(
                list(set([sel.row() for sel in self.selectedIndexes()]))
            )

            while len(selected_rows) > 0:
                target = selected_rows.pop()
                self._data_model.removeRows(target)
        else:
            alert("Warning", "Can't remove the only remaining row.")

    def removeSelectedColumns(self):
        Ncolumns = self._data_model.df.shape[1]

        if Ncolumns > 1:
            selected_columns = sorted(
                list(set([sel.column() for sel in self.selectedIndexes()]))
            )

            while len(selected_columns) > 0:
                target = selected_columns.pop()
                self._data_model.removeColumns(target)
        else:
            alert("Warning", "Can't remove the only remaining column.")

    def exclude_selected_rows(self):
        selected_rows = list(
            set(
                [
                    self._data_model.df.index[sel.row()]
                    for sel in self.selectedIndexes()
                ]
            )
        )

        for row in selected_rows:
            self._data_model.excluded_index.append(row)

    def remove_exlusion(self):
        selected_rows = list(
            set(
                [
                    self._data_model.df.index[sel.row()]
                    for sel in self.selectedIndexes()
                ]
            )
        )

        for row in selected_rows:
            if row in self._data_model.excluded_index:
                self._data_model.excluded_index.remove(row)

    def prototype_set_filter(self):
        column_names = self._data_model.df.columns
        # check if table still has excluded data
        Nexcluded = len(self._data_model.excluded_index)
        if Nexcluded > 0:
            alert("Warning", "please remove excluded data before doing filtering.")
        else:
            filter_dialog = FilterDialog(column_names, parent=self)
            filter_dialog.show()
            filter_dialog.exec_()

    def getVisibleData(self):

        data = self._data_model.df
        excluded = self._data_model.excluded_index

        return data.drop(excluded, axis=0)


class DataFrameModel(QAbstractTableModel):
    def __init__(self):
        super(DataFrameModel, self).__init__()
        self._df = pandas.DataFrame()
        self.Ncoladded = 0
        self.excluded_index = []
        self.filters = []
        # internal dataframe for filtered dataframe
        self._filtered = pandas.DataFrame()

    def setDataFrame(self, dataframe):
        # dataframe initialization
        self.modelAboutToBeReset.emit()
        self._df = dataframe
        self.modelReset.emit()

    @property
    def df(self):
        """ use this proper to access `visible` table """
        if len(self.filters) == 0:
            return self._df
        else:
            return self._filtered

    @df.setter
    def df(self, dataframe):
        if len(self.filters) == 0:
            self.modelAboutToBeReset.emit()
            self._df = dataframe
            self.modelReset.emit()
        else:
            self.modelAboutToBeReset.emit()
            self._filtered = dataframe
            self.modelReset.emit()

    # table display functions
    def data(self, index, role):
        """ this method determines how data is presented in the table """
        data_id = self.df.index[index.row()]

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

        elif role == Qt.BackgroundRole and data_id in self.excluded_index:
            # mark excluded data by coloring the background gray
            return QBrush(Qt.gray)

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

    def excludeRow(self, row_index):
        if row_index not in self.excluded_row:
            self.excluded_index.append(row_index)

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
            # get indices from visible table
            row = index.row()
            column = index.column()

            # map to actual internal dataframe index
            glob_row = self.df.index[row]
            glob_col = self.df.columns[column]

            if value == "":
                return False

            # try to convert data to int or float
            if '.' in value:
                # convert to float
                try:
                    value = float(value)
                except ValueError:
                    # keep as string
                    pass

            else:
                # convert to integer
                try:
                    value = int(value)
                except ValueError:
                    pass

            # assign data to internal data frame
            self._df.loc[glob_row, glob_col] = value
            # assign data to visible data frame (self.filtered is a copy!)
            self.df.loc[glob_row, glob_col] = value

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

    def addFilter(self, expression):
        if expression not in self.filters:
            self.filters.append(expression)

    def filterData(self):
        # combine expressions
        if len(self.filters) > 1:
            combinedexpr = " & ".join([filt for filt in self.filters])
        elif len(self.filters) == 1:
            combinedexpr = self.filters[0]
        elif len(self.filters) == 0:
            # dont do any queries
            self.layoutAboutToBeChanged.emit()
            self.df = self._df
            self.layoutChanged.emit()
            return None
        self.layoutAboutToBeChanged.emit()
        # do filtering only on internal original dataframe
        self.df = self._df.query(combinedexpr)
        self.layoutChanged.emit()

    def resetFilter(self):
        self.layoutAboutToBeChanged.emit()
        self.filters = []
        self._filtered = pandas.DataFrame()
        self.layoutChanged.emit()

    def resetExcluded(self):
        self.excluded_index = []

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

        self.endInsertRows()

        return True

    def removeRows(self, position, count=1, index=QModelIndex()):
        """ remove rows, only count=1 cases has been tested """
        self.beginRemoveRows(QModelIndex(), position, position + count - 1)
        self.df = self.df.drop(self.df.index[position], axis=0)
        self.df.index = [i for i in range(self.df.shape[0])]
        self.endRemoveRows()

        return True

    def insertColumns(self, position, count=1, index=QModelIndex()):
        """ function to remove column(s), new columns are named NewColumn# """
        # to keep track of naming

        self.beginInsertColumns(QModelIndex(), position, position + count - 1)

        for c in range(count):
            self.Ncoladded += 1
            # insert is an in-place operation
            self.df.insert(position, f"NewColumn{self.Ncoladded}", value=nan)

        self.endInsertColumns()

        return True

    def removeColumns(self, position, count=1, index=QModelIndex()):
        """ remove columns, only count=1 cases have been tested """
        self.beginRemoveColumns(QModelIndex(), position, position + count - 1)
        self.df.drop(self.df.columns[position], axis=1, inplace=True)
        self.endRemoveColumns()

        return True
