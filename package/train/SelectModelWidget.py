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

from package.train.models.SkModelDialog import SkModelDialog
from package.utils.catutils import exceptionWarning

BASEMODEL_DIR = "./package/data/basemodels"
BASETFIDF_DIR = "./package/data/featureextractors/tfidf.json"
class SelectModelWidget(QTabWidget):
    def __init__(self, parent=None):
        super(SelectModelWidget, self).__init__(parent)
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
            row = 0
            for filename in os.listdir(BASEMODEL_DIR):
                with open(os.path.join(BASEMODEL_DIR, filename), 'r') as f:
                    print("basemodel file being loaded", filename)
                    model_data = json.load(f)
                    model_dialog = SkModelDialog(self, model_data, tfidf_data)
                    btn = QPushButton(model_data['model_class'])
                    btn.clicked.connect(lambda: self.openDialog(model_dialog))
                    self.model_select_grid.addWidget(btn, row, 0)
                    self.model_dialogs.append(model_dialog)
                    self.model_dialog_btns.append(btn)
                    row += 1
        except OSError as ose:
            exceptionWarning('Error opening base model config files!', ose)
        except Exception as e:
            exceptionWarning('Error occured.', e)

    def openDialog(self, dialog):
        dialog.saveParams()