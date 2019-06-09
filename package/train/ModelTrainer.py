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

import pandas as pd

from sklearn.pipeline import make_pipeline

import package.utils.training_utils as tu

class ModelTrainer(QThread):
    """
    QThread tasked with running all model training/tuning.  
    This could potentially take days to complete.
    """
    training_complete = Signal(pd.DataFrame)

    def __init__(self, selected_models, version_directory, 
                 training_eval_params, training_data, **kwargs
                ):
        super(ModelTrainer, self).__init__()
        self.version_directory = version_directory
        self.selected_models = selected_models
        print(self.selected_models)
        self.training_eval_params = training_eval_params
        print(json.dumps(self.training_eval_params, indent=2))
        self.training_data = training_data
        self.kwargs = kwargs

    
    def run(self):
        pass
