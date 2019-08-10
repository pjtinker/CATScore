'''
CATEvaluate module contains the all functionality associated with the training
 and tuning of machine-learning models.

@author pjtinker
'''

import sys
import importlib
import traceback
import argparse
import logging
import json
import os

import pandas as pd

from PyQt5.QtCore import (Qt, pyqtSlot, pyqtSignal)
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QDialog, QHeaderView, QAction,
                               QMainWindow, QSizePolicy, QProgressBar, QWidget,
                               QVBoxLayout, QFormLayout, QGroupBox, QLineEdit,
                               QLabel, QDialogButtonBox, QMessageBox, QPushButton)

from package.utils.catutils import exceptionWarning
from package.evaluate.ScoreWidget import ScoreWidget

VERSION_BASE_DIR = "./package/data/versions"
DEFAULT_QUESTION_LABELS = ['Q1', 'Q2', 'Q3', 'Q4', 'Q6',
                    'Q7', 'Q9', 'Q11', 'Q14', 'Q15']
class CatEvaluate(QMainWindow):
    """ The central widget for the training component of CATScore
        Most of the functionality is contained in this class
    """
    def __init__(self, parent=None):
        super(CatEvaluate, self).__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.title = 'CAT Evaluate'
        self.setWindowTitle(self.title)
        geometry = QApplication.desktop().availableGeometry(self)
        parent_left = self.parent().geometry().left()
        parent_top = self.parent().geometry().top()
        self.setGeometry(parent_left, parent_top, geometry.width() * 0.45, geometry.height() * 0.82)
        self.statusBar().showMessage('Cut me, Mick!')
        self.progressBar = QProgressBar()
        self.progressBar.setGeometry(30, 40, 200, 25)
        self.progressBar.setFormat("Idle")
        self.progressBar.setTextVisible(True)
        self.file_menu = self.menuBar().addMenu('&File')

        # self.version_menu = self.menuBar().addMenu('&Version')
        self.statusBar().addPermanentWidget(self.progressBar)
        self.score_widget = ScoreWidget(self)
        # self.version_widget.version_created.connect(self.score_widget.model_widget.add_new_version)
        self.score_widget.data_predictor.comms.update_progressbar.connect(self.update_progress_bar)
        # self.score_widget.model_widget.update_progressbar.connect(self.update_progress_bar)
        self.setCentralWidget(self.score_widget)

    @pyqtSlot(int, bool)
    def update_progress_bar(self, val, pulse):
        if pulse:
            self.progressBar.setRange(0,0)
        else:
            self.progressBar.setRange(0, 1)
        self.progressBar.setValue(val)

if __name__ == "__main__":
    import sys
    # Qt Application
    # app = QApplication(sys.argv)
    # # app.setStyle('Fusion')
    # window = CatEvaluate()
    # window.show()
    # sys.exit(app.exec_())