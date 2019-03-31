from PySide2.QtCore import (QAbstractTableModel, QDateTime, QModelIndex,
                            Qt, QTimeZone, QByteArray, Slot, SIGNAL)
from PySide2.QtWidgets import (QAction, QGroupBox, QMessageBox, QCheckBox, 
                                QTabWidget, QComboBox,
                                QApplication, QLabel, QFileDialog, QHBoxLayout, 
                                QVBoxLayout, QGridLayout, QHeaderView, QScrollArea, 
                                QSizePolicy, QTableView, QWidget, QPushButton)
import os
import json
from collections import OrderedDict
import pkg_resources
import traceback 
from functools import partial
import logging 

import pandas as pd
from addict import Dict

from package.train.models.SkModelDialog import SkModelDialog
from package.train.models.TfModelDialog import TfModelDialog
from package.utils.catutils import exceptionWarning

BASE_SK_MODEL_DIR = "./package/data/basemodels"
BASE_TF_MODEL_DIR = "./package/data/tensorflowmodels"
BASE_TFIDF_DIR = "./package/data/featureextractors/tfidf.json"
class SelectModelWidget(QTabWidget):
    def __init__(self, parent=None):
        super(SelectModelWidget, self).__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.parent = parent
        self.selected_version = 'default'

        self.sklearn_model_dialogs = []
        self.sklearn_model_dialog_btns = []
        self.sklearn_model_checkboxes = []

        self.tf_model_dialogs = []
        self.tf_model_dialog_btns = []
        self.tf_model_checkboxes = []

        self.main_layout = QHBoxLayout()
        self.left_column = QVBoxLayout()
        self.right_column = QVBoxLayout()
        self.version_hbox = QHBoxLayout()

        self.skmodel_groupbox = QGroupBox("Sklearn Models")
        self.tf_model_groupbox = QGroupBox("Tensorflow Models")
        self.tf_model_grid = QGridLayout()
        self.sklearn_model_grid = QGridLayout()
        self.skmodel_groupbox.setLayout(self.sklearn_model_grid)
        self.tf_model_groupbox.setLayout(self.tf_model_grid)

        self.setupUi()
        self.left_column.addLayout(self.version_hbox)
        # self.version_hbox.addStretch()
        self.left_column.addWidget(self.skmodel_groupbox)
        self.left_column.addStretch()
        self.right_column.addWidget(self.tf_model_groupbox)
        self.right_column.addStretch()
        self.main_layout.addLayout(self.left_column)
        # self.main_layout.addStretch()
        self.main_layout.addLayout(self.right_column)
        self.main_layout.addStretch()
        self.setLayout(self.main_layout)

    def setupUi(self):
        self.version_selection = QComboBox(objectName='version_select')
        self.version_selection.addItem('Default', None)
        available_versions = os.listdir(".\\package\\data\\versions")
        for version in available_versions:
            self.version_selection.addItem(version, os.path.join('.\\package\\data\\versions', version))
        self.version_selection.currentIndexChanged.connect(lambda x, y=self.version_selection: 
                                                            print(y.currentData())
                                                            )
        self.version_hbox.addWidget(self.version_selection)
        try:
            with open(BASE_TFIDF_DIR, 'r') as f:
                tfidf_data = json.load(f)
        except IOError as ioe:
            self.logger.error("Error loading base TFIDF params", exc_info=True)
            exceptionWarning('Error occurred while opening base TFIDF vectorizer.', repr(ioe))
        try:
            row = 0
            for filename in os.listdir(BASE_SK_MODEL_DIR):
                with open(os.path.join(BASE_SK_MODEL_DIR, filename), 'r') as f:
                    print("Loading model:", filename)
                    model_data = json.load(f)
                    model_dialog = SkModelDialog(self, model_data, tfidf_data)
                    btn = QPushButton(model_data['model_class'])
                    #~ Partial allows the connection of dynamically generated
                    #~ QObjects
                    btn.clicked.connect(partial(self.openDialog, model_dialog))
                    chkbox = QCheckBox()
                    self.sklearn_model_grid.addWidget(chkbox, row, 0)
                    self.sklearn_model_grid.addWidget(btn, row, 1)
                    self.sklearn_model_dialogs.append(model_dialog)
                    self.sklearn_model_dialog_btns.append(btn)
                    self.sklearn_model_checkboxes.append(chkbox)
                    row += 1
        except OSError as ose:
            self.logger.error("OSError opening Scikit model config files", exc_info=True)
            exceptionWarning('OSError opening Scikit model config files!', ose)
        except Exception as e:
            self.logger.error("Error opening Scikit model config files", exc_info=True)
            exceptionWarning('Error occured.', e)
            tb = traceback.format_exc()
            print(tb)

        try:
            row = 0
            for filename in os.listdir(BASE_TF_MODEL_DIR):
                with open(os.path.join(BASE_TF_MODEL_DIR, filename), 'r') as f:
                    print("Loading model:", filename)
                    model_data = json.load(f)
                    model_dialog = TfModelDialog(self, model_data)
                    btn = QPushButton(model_data['model_class'])
                    btn.clicked.connect(partial(self.openDialog, model_dialog))
                    chkbox = QCheckBox()
                    self.tf_model_grid.addWidget(chkbox, row, 0)
                    self.tf_model_grid.addWidget(btn, row, 1)
                    self.tf_model_dialogs.append(model_dialog)
                    self.tf_model_dialog_btns.append(btn)
                    self.tf_model_checkboxes.append(chkbox)
                    row += 1
        except OSError as ose:
            self.logger.error("OSError opening Tensorflow model config files", exc_info=True)
            exceptionWarning('Error opening Tensorflow model config files!', ose)
        except Exception as e:
            self.logger.error("Error opening Scikit model config files", exc_info=True)
            exceptionWarning('Error occured.', e)
            tb = traceback.format_exc()
            print(tb)

    def openDialog(self, dialog):
        dialog.saveParams()

    