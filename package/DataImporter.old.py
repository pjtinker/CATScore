from PySide2.QtCore import (QAbstractTableModel, QDateTime, QModelIndex,
                            Qt, QTimeZone, Slot)
from PySide2.QtGui import QColor, QPainter
from PySide2.QtWidgets import (QAction, QApplication, QFileDialog, QHBoxLayout, QVBoxLayout, QHeaderView,
                               QMainWindow, QSizePolicy, QTableView, QWidget, QPushButton)
from PySide2.QtCharts import QtCharts

import pandas as pd
import glob
from chardet.universaldetector import UniversalDetector

from DataframeTableModel import DataframeTableModel

class DataImporter(QWidget):
    def __init__(self, parent=None):
        super(DataImporter, self).__init__(parent)
        self.model = DataframeTableModel(self)
        self.table_view = QTableView()
        self.table_view.setModel(self.model)
        self.table_view.setSortingEnabled(True)

        self.main_layout = QVBoxLayout()
        size = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        size.setHorizontalStretch(1)
        self.table_view.setSizePolicy(size)
        self.main_layout.addWidget(self.table_view)

        self.open_file_button = QPushButton('Add', self)
        self.open_file_button.clicked.connect(lambda: self.openFile())
        self.main_layout.addWidget(self.open_file_button)
        self.setLayout(self.main_layout)

    def openFile(self):
        file_name, filter = QFileDialog.getOpenFileName(self)
        if file_name:
            self.loadFile(file_name)

    def loadFile(self, f_name):
        print("Attempting to open {}".format(f_name))
        detector = UniversalDetector()
        for line in open(f_name, 'rb'):
            detector.feed(line)
            if detector.done: break
        detector.close()
        print("chardet determined encoding type to be {}".format(detector.result['encoding']))
        data = pd.read_csv(f_name, encoding=detector.result['encoding'])
        
        self.model.loadData(data)