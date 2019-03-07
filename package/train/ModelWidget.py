from PySide2.QtCore import (QAbstractTableModel, QDateTime, QModelIndex,
                            Qt, QTimeZone, QByteArray, Slot, SIGNAL)
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

from package.train.models.ModelDialog import ModelDialog
from package.utils.catutils import exceptionWarning

BASEMODEL_DIR = "./package/data/basemodels"
BASETFIDF_DIR = "./package/data/featureextractors/tfidf.json"
class ModelWidget(QTabWidget):
    def __init__(self, parent=None):
        super(ModelWidget, self).__init__(parent)
        self.parent = parent
        self.model_dialogs = []
        self.model_dialog_btns = []

        self.main_layout = QHBoxLayout()
        self.model_select_grid = QGridLayout()
        self.setupUi()

        self.main_layout.addLayout(self.model_select_grid)
        self.setLayout(self.main_layout)
    def setupUi(self):
        try:
            with open(BASETFIDF_DIR, 'r') as f:
                tfidf_data = json.load(f)
        except IOError as ioe:
            exceptionWarning('Error occurred while opening base TFIDF vectorizer.', repr(ioe))
        try:
            for filename in os.listdir(BASEMODEL_DIR):
                with open(os.path.join(BASEMODEL_DIR, filename), 'r') as f:
                    model_data = json.load(f)
                    model_dialog = ModelDialog(self, model_data, tfidf_data)
                    btn = QPushButton(model_data['model_class'])
                    btn.clicked.connect(lambda: self.openDialog(model_dialog))
                    self.model_select_grid.addWidget(btn, 0, 0)
                    self.model_dialogs.append(model_dialog)
                    self.model_dialog_btns.append(btn)
        except OSError as ose:
            exceptionWarning('Error opening base model config files!', ose)
        except Exception as e:
            exceptionWarning('Error occured.', e)

    def openDialog(self, dialog):
        dialog.saveParams()