"""QFormLayou for model training parameters
"""
from PySide2.QtCore import Signal, Slot
from PySide2.QtWidgets import (QPushButton, QApplication, QHBoxLayout, QVBoxLayout, QFormLayout, 
                               QGroupBox, QWidget, QLineEdit, QGridLayout,
                               QDialog, QSpinBox, QDialogButtonBox, QComboBox, 
                               QDoubleSpinBox, QSizePolicy, QLabel)
import json
import re
import importlib 
import traceback
import inspect
import logging
import os

class TrainParamsWidget(QFormLayout):
    """
    QFormLayout holding several general parameters for model training.
    """