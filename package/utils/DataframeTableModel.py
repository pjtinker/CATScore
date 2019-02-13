from PySide2.QtCore import (QAbstractTableModel, QDateTime, QModelIndex,
                            Qt, QTimeZone, Slot)
from PySide2.QtGui import QColor, QPainter
from PySide2.QtWidgets import (QAction, QApplication, QHBoxLayout, QHeaderView,
                               QMainWindow, QSizePolicy, QTableView, QWidget, QTabWidget)
from PySide2.QtCharts import QtCharts


import pandas as pd
import numpy as np

"""
@author: pjtinker
"""

"""QTableView based on pandas DataFrame.
Provides functionality to create a QTableView using a pandas DataFrame as the 
underlying data structure.
Currently, this class provides display only.  Potentially table editing and sorting
will be added.
"""

class DataframeTableModel(QAbstractTableModel):
    
    def __init__(self, parent=None):
        QAbstractTableModel.__init__(self)    
        self.color = None
        self._df = pd.DataFrame()
        self.header = []

    def loadData(self, data):
        """Load data into dataframe.
        Currently data is passed into the function
        Perhaps adding it internally would make better sense?
        """
        if data is None:
            return
        self._df = data
        self.header = self._df.columns
        ## layoutChanged refreshes the QTableView with new data.  
        self.layoutChanged.emit()

    def rowCount(self, parent=QModelIndex()):
        return self._df.shape[0]

    def columnCount(self, parent=QModelIndex()):
        return self._df.shape[1]

    def headerData(self, section, orientation, role):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except(IndexError, 
            ):
                return None
        elif orientation == Qt.Vertical:
            try:
                return self._df.index.tolist()[section]
            except(IndexError, ):
                return None

    def data(self, index, role=Qt.DisplayRole):
        column = index.column()
        row = index.row()
        if role != Qt.DisplayRole:
            return None
        if not index.isValid():
            return None
        return str(self._df.ix[row, column])

    '''FIXME: Not sure I'll allow the data to be editable.  
        This is here just in case we do.  
    '''
    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            # PyQt4 gets a QVariant
            value = value.toPyObject()
        else:
            # PySide gets an unicode
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.at[row, col] = value
        return True

    ##FIXME: sorting is broken
    # def sort(self, Ncol, order):
    #     '''Sort table by given column number'''
    #     self.layoutAboutToBeChanged.emit()
    #     self._df = self._df.sort_values(self.header[Ncol], 
    #                                     ascending=order == Qt.AscendingOrder)
    #     self.layoutChanged.emit()

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemIsEnabled
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable 

        return flags