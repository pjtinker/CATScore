from PyQt5.QtCore import (Qt)
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import (QPushButton, QApplication, QHBoxLayout, QVBoxLayout, QFormLayout, 
                               QGroupBox, QWidget, QLineEdit, QGridLayout, QMessageBox,
                               QDialog, QSpinBox, QDialogButtonBox, QComboBox, 
                               QDoubleSpinBox, QSizePolicy, QLabel, QPlainTextEdit)
import os
import json
import inspect
import importlib
import logging
"""
Utility classes for CATScore

@author pjtinker
"""
class CATEncoder(json.JSONEncoder):
    def default(self, obj):
        if inspect.isclass(obj):
            return{
                "type" : "__class__",
                "module" : obj.__module__,
                "name" : obj.__name__
            }
        if inspect.isfunction(obj):
            return{
                "type" : "__function__",
                "module" : obj.__module__,
                "name" : obj.__name__
            }
        return json.JSONEncoder.default(self, obj)




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



def cat_decoder(obj):
    if "score_func" in obj:
        module = importlib.import_module(obj['score_func']['module'])
        return {"score_func" : getattr(module, obj['score_func']['name'])}

    if "ngram_range" in obj:
        return {"ngram_range" : (obj['ngram_range']) }

    return obj