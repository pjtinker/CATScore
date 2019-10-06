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

from package.evaluate.PredictWidget import PredictWidget
from package.evaluate.EvaluateWidget import EvaluateWidget

class ScoreWidget(QTabWidget):
    def __init__(self, parent=None):
        super(ScoreWidget, self).__init__(parent)
        self.parent = parent
        # FIXME: reset statusbar when tabs are changed
        self.currentChanged.connect(lambda: self.update_statusbar('Ready'))
        self.data_predictor = PredictWidget(self)
        self.addTab(self.data_predictor, 'Predict')      
        self.setTabEnabled(1, True)
        self.data_predictor.comms.update_statusbar.connect(self.update_statusbar)
        self.evaluator = EvaluateWidget(self)
        self.addTab(self.evaluator, 'Evaluate')
        self.setTabEnabled(2, True)

    @pyqtSlot(int, bool)
    def setTab(self, tab, state):
        self.setTabEnabled(tab, state)


    @pyqtSlot(str)
    def update_statusbar(self, msg):
        self.parent.statusBar().showMessage(msg)
        self.parent.repaint()

    def setupTabs(self):
        pass
