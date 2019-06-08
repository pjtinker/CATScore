"""
QThread for model training
"""
from PySide2.QtCore import (Qt, Slot, Signal, QObject, QThread)

import json
import re
import importlib 
import traceback
import inspect
import logging
import os

class ModelTrainer(QThread):
    
    def __init__(self, version, training_cols, training_params, train_data, **kwargs):
        super(self.ModelTrainer, self).__init__()
        self.version = version
        self.training_cols = training_cols
        self.training_params = training_params
        self.training_data = training_data
        self.kwargs = kwargs

