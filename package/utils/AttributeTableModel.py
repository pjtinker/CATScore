from PyQt5.QtCore import (QAbstractTableModel, QDateTime, QModelIndex,
                            Qt, pyqtSlot, pyqtSignal)
from PyQt5.QtWidgets import (QCheckBox, QSizePolicy, QWidget)

from package.utils.DataframeTableModel import DataframeTableModel
import pandas as pd
import numpy as np

"""
@author: pjtinker
"""

"""QTableView used for question selection.  Allows users to check
    the questions they would like to include in the subsequent analyses.  
"""

class AttributeTableModel(DataframeTableModel):
    def __init__(self, parent=None):
        DataframeTableModel.__init__(self)
        self.checklist = []

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        column = index.column()
        row = index.row()
        if role == Qt.DisplayRole:
            return str(self._df.ix[row, column])
        elif role == Qt.CheckStateRole and column == 0:
            if self.checklist[row]:
                return Qt.Checked
            else:
                return Qt.Unchecked

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemIsEnabled
        if index.column() == 0:
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable 
        else:
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable

        return flags

    def loadData(self, data, include_labels=True):
        if data is None:
            return
        if include_labels:
            if len(data) % 2 != 0:
                raise IndexError('Invalid number of parameters for question/label pairs.')
            it = iter(data)
            data_tuples = list(zip(it, it))
            self._df = pd.DataFrame(data_tuples, columns=['Text', 'Label'])
        else:
            self._df = pd.DataFrame(data, columns=['Text'])
        self.checklist = [False for _ in range(self.rowCount())]
        # print(self._df.head())
        self.layoutChanged.emit()

    def setCheckboxes(self, truth=True):
        """Select/deselect all questions
        """
        self.checklist = [truth for _ in range(self.rowCount())]
        self.layoutChanged.emit()


    def setData(self, index, value, role):
        if not index.isValid():
            return None
        col = index.column()
        row = index.row()
        if role == Qt.CheckStateRole:
            #FIXME: Perhaps a different configuration that does not use a list?
            # Append actual column names?  Use dict?
            self.checklist[row] = not self.checklist[row]
            return True
        else:
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.at[row, col] = value
        return True

    def getChecklist(self):
        selected_cols = []
        for idx, truth in enumerate(self.checklist):
            if truth:
                selected_cols.extend(self._df.iloc[idx].values)
        return selected_cols

        