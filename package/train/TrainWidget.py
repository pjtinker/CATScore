from PySide2.QtCore import (QAbstractTableModel, QDateTime, QModelIndex,
                            Qt, QTimeZone, QByteArray, Slot, SIGNAL)
from PySide2.QtGui import QMovie
from PySide2.QtWidgets import (QAction, QGroupBox, QMessageBox, QCheckBox, 
                                QTabWidget,
                                QApplication, QLabel, QFileDialog, QHBoxLayout, 
                                QVBoxLayout, QGridLayout, QHeaderView, QScrollArea, 
                                QSizePolicy, QTableView, QWidget, QPushButton)
import os
import json
from collections import OrderedDict
import pkg_resources
import pandas as pd
from addict import Dict

from package.utils.DataLoader import DataLoader
from package.train.ModelWidget import ModelWidget

class TrainWidget(QTabWidget):
    def __init__(self, parent=None):
        super(TrainWidget, self).__init__(parent)
        self.parent = parent
        self.data_loader = DataLoader(self)
        self.model_widget = ModelWidget(self)
        self.addTab(self.data_loader, 'Load Data')      
        self.addTab(self.model_widget, 'Model Selection')
        self.setTabEnabled(1, False)
        self.data_loader.data_load.connect(self.loadData)

    @Slot(pd.DataFrame)
    def loadData(self, data):
        #FIXME: Copy data or keep reference?
        self.full_training_set = data
        self.setTabEnabled(1, True)
        self.parent.statusBar().showMessage('Training data loaded.')

    def setupImportTab(self):
        self.main_layout = QHBoxLayout()
        self.left_column = QVBoxLayout()
        self.right_column = QVBoxLayout()

        self.left_column.addWidget()
        self.left_column.addStretch()
        self.main_layout.addLayout(self.left_column)
        self.main_layout.addStretch()
        
        self.main_layout.addLayout(self.right_column)
        self.setLayout(self.main_layout)

    def setupTabs(self):
        pass
