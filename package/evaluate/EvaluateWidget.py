import json
import logging
import os
import traceback
import time
from collections import OrderedDict
from functools import partial
import hashlib

import pandas as pd
import pkg_resources
from PyQt5.QtCore import QObject, Qt, QThread, QThreadPool, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtWidgets import (QAction, QButtonGroup, QCheckBox, QComboBox,
                             QDoubleSpinBox, QFileDialog, QFormLayout,
                             QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                             QMessageBox, QPushButton, QRadioButton,
                             QScrollArea, QSizePolicy, QSpinBox, QTabWidget,
                             QVBoxLayout, QPlainTextEdit, QWidget, QTableView)

from package.utils.catutils import exceptionWarning, clearLayout
from package.utils.DataframeTableModel import DataframeTableModel
from package.utils.AttributeTableModel import AttributeTableModel
from package.utils.GraphWidget import GraphWidget


class Communicate(QObject):
    pass


class EvaluateWidget(QWidget):
    def __init__(self, parent=None):
        super(EvaluateWidget, self).__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.parent = parent
        self.comms = Communicate()

        self.prediction_data = pd.DataFrame()
        self.open_file_button = QPushButton('Load CSV', self)
        self.open_file_button.clicked.connect(lambda: self.open_file())

        self.main_layout = QHBoxLayout()
        self.left_column = QVBoxLayout()
        self.right_column = QVBoxLayout()

        # ~ Available question column view
        self.available_column_view = QTableView()
        self.available_column_view.setMinimumHeight(322)
        self.available_column_view.setMaximumWidth(214)
        self.available_column_view.setSelectionMode(QTableView.SingleSelection)
        self.available_column_view.setSelectionBehavior(QTableView.SelectRows)
        self.available_column_model = AttributeTableModel()
        self.available_column_view.setModel(self.available_column_model)
        selection = self.available_column_view.selectionModel()
        selection.selectionChanged.connect(
            lambda x: self.display_selected_rows(x))

        self.left_column.addWidget(self.open_file_button)
        self.left_column.addWidget(self.available_column_view)

        self.setLayout(self.main_layout)
