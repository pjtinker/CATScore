from PyQt5.QtCore import (Qt)
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import (QPushButton, QApplication, QHBoxLayout, QVBoxLayout, QFormLayout, 
                               QGroupBox, QWidget, QLineEdit, QGridLayout, QMessageBox,
                               QDialog, QSpinBox, QDialogButtonBox, QComboBox, 
                               QDoubleSpinBox, QSizePolicy, QLabel)
import os
import json
import inspect

"""
Utility classes for CATScore

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


class CATEncoder(json.JSONEncoder):
    def default(self, obj):
        if inspect.isclass(obj):
            return obj.__name__
        if inspect.isfunction(obj):
            return '.'.join([obj.__module__, obj.__name__])
        return json.JSONEncoder.default(self, obj)