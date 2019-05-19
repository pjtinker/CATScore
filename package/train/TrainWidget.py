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
# from addict import Dict

from package.utils.DataLoader import DataLoader
from package.train.SelectModelWidget import SelectModelWidget

class TrainWidget(QTabWidget):
    def __init__(self, parent=None):
        super(TrainWidget, self).__init__(parent)
        self.parent = parent
        # FIXME: reset statusbar when tabs are changed
        # self.currentChanged.connect(self.updateStatusBar(' '))
        self.data_loader = DataLoader(self)
        self.model_widget = SelectModelWidget(self)
        self.addTab(self.data_loader, 'Load Data')      
        self.addTab(self.model_widget, 'Model Selection')
        self.setTabEnabled(1, True)
        self.data_loader.data_load.connect(self.setTab)
        self.data_loader.update_statusbar.connect(self.updateStatusBar)

    @Slot(int, bool)
    def setTab(self, tab, state):
        self.setTabEnabled(tab, state)


    @Slot(str)
    def updateStatusBar(self, msg):
        self.parent.statusBar().showMessage(msg)
        self.parent.repaint()

    def setupTabs(self):
        pass
