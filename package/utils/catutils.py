from PySide2.QtCore import (Qt)
from PySide2.QtGui import QMovie
from PySide2.QtWidgets import (QAction, QMessageBox, QCheckBox, QApplication, QLabel, QFileDialog, QHBoxLayout, QVBoxLayout, 
                               QGridLayout, QHeaderView, QSizePolicy, QTableView, QWidget, QPushButton)
import os

"""Utility classes for CATScore

@author pjtinker
"""

def exceptionWarning(exceptionText, exception=None, title='Warning'):
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setText(exceptionText)
    if exception:
        msg_box.setInformativeText(repr(exception))
    msg_box.exec_()

def clearLayout(layout):
    while layout.count():
        child = layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()