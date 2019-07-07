from PyQt5.QtCore import (QAbstractTableModel, QDateTime, QModelIndex,
                            Qt, QTimeZone, QByteArray, pyqtSlot, pyqtSignal)
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import (QAction, QGroupBox, QMessageBox, QCheckBox, 
                                QTabWidget,
                                QApplication, QLabel, QFileDialog, QHBoxLayout, 
                                QVBoxLayout, QGridLayout, QHeaderView, QScrollArea, 
                                QSizePolicy, QTableView, QWidget, QMenuBar, QPushButton)
import os
import json
from collections import OrderedDict
import pkg_resources
import pandas as pd
# from addict import Dict

from package.utils.DataLoader import DataLoader
from package.train.SelectModelWidget import SelectModelWidget

class TrainWidget(QTabWidget):
    def __init__(self, parent=None):
        super(TrainWidget, self).__init__(parent)
        self.parent = parent
        # FIXME: reset statusbar when tabs are changed
        self.currentChanged.connect(lambda: self.update_statusbar('Ready'))
        self.data_loader = DataLoader(self)
        self.model_widget = SelectModelWidget(self)
        self.addTab(self.data_loader, 'Load Data')      
        self.addTab(self.model_widget, 'Model Selection')
        self.setTabEnabled(1, True)
        self.data_loader.data_load.connect(self.model_widget.load_data)
        self.data_loader.update_statusbar.connect(self.update_statusbar)

    @pyqtSlot(int, bool)
    def setTab(self, tab, state):
        self.setTabEnabled(tab, state)


    @pyqtSlot(str)
    def update_statusbar(self, msg):
        self.parent.statusBar().showMessage(msg)
        self.parent.repaint()

    def setupTabs(self):
        pass
