from PySide2.QtCore import (QAbstractTableModel, QDateTime, QModelIndex,
                            Qt, QTimeZone, QByteArray, Slot)
from PySide2.QtGui import QMovie
from PySide2.QtWidgets import (QAction, QGroupBox, QMessageBox, QCheckBox, QApplication, QLabel, QFileDialog, QHBoxLayout, QVBoxLayout, 
                               QGridLayout, QHeaderView, QScrollArea, QSizePolicy, QTableView, QWidget, QPushButton)
import os
import pandas as pd

from package.utils.DataImporter import DataImporter
from package.train.models.svc import SVC

class PreprocessingTab(QWidget):
    def __init__(self, parent=None):
        super(PreprocessingTab, self).__init__(parent)
        self.importer = DataImporter(self)

        self.main_layout = QHBoxLayout()
        self.left_column = QVBoxLayout()
        self.right_column = QVBoxLayout()

        self.left_column.addWidget(self.importer)
        self.left_column.addStretch()
        self.main_layout.addLayout(self.left_column)

        self.svc_button = QPushButton("SVC")
        svc_params = {
                        "clf__C": 1,
                        "clf__class_weight": None,
                        "clf__kernel": "linear",
                        "clf__tol": 0.001,
                        "clf__cache_size": 200,
                        "tfidf__max_df": 0.5,
                        "tfidf__min_df": 2,
                        "tfidf__ngram_range": [
                            1,
                            2
                        ],
                        "tfidf__stop_words": None,
                        "tfidf__strip_accents": "unicode",
                        "tfidf__use_idf": False
                    }
        self.svc_button.clicked.connect(lambda: self._openDialog(SVC(svc_params)))
        self.right_column.addWidget(self.svc_button)
        self.main_layout.addLayout(self.right_column)
        self.setLayout(self.main_layout)

    def _openDialog(self, dialog):
        dialog.saveParams()        