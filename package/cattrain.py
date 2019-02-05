'''
CATTrain module contains the all functionality associated with the training
of new machine-learning models.

@author pjtinker
'''

import sys
import argparse
import pandas as pd

from PySide2.QtCore import (QAbstractTableModel, QDateTime, QModelIndex,
                            Qt, QTimeZone, Slot)
from PySide2.QtGui import QColor, QPainter
from PySide2.QtWidgets import (QAction, QApplication, QHBoxLayout, QHeaderView,
                               QMainWindow, QSizePolicy, QTableView, QWidget, QTabWidget)
from PySide2.QtCharts import QtCharts

from package.DataImporter import DataImporter

class CatTrain(QMainWindow):
    def __init__(self, parent=None):
        super(CatTrain, self).__init__(parent)

        self.current_file = ''
        self.title = 'CAT Train'
        self.left = 0
        self.top = 0
        self.width = 500
        self.height = 400
        self.setWindowTitle(self.title)
        geometry = QApplication.desktop().availableGeometry(self)
        self.setGeometry(0, 0, geometry.width() * 0.8, geometry.height() * 0.7)
        # self.setGeometry(0, 0, 600, 400)

        self.tabs = QTabWidget()
        self.import_answers_tab = DataImporter()
        self.select_models_tab = QWidget()
        # self.setCentralWidget(self.widget)
        self.tabs.resize(500, 400)

        self.tabs.addTab(self.import_answers_tab, 'Import Answer Set')
        self.tabs.addTab(self.select_models_tab, 'Select Models')
 
        self.setCentralWidget(self.tabs)

    def closeEvent(self, event):
        print("closeEvent fired")

