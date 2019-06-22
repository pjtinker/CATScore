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
        print("Initializing ModelTrainer...")
        self.version_directory = version_directory
        print(self.version_directory)
        self.selected_models = selected_models
        print(self.selected_models)
        self.training_eval_params = training_eval_params
        print(json.dumps(self.training_eval_params, indent=2))
        self.training_data = training_data
        print(self.training_data.head())
        self.kwargs = kwargs

    
    def run(self):
        # Run thru enumeration of columns.  The second argument in enumerate
        # tells python where to begin the idx count.  Here 1 for our offset
        for col_idx, col in enumerate(self.training_data.columns, 1): 
            if col.endswith('_Text'):
                print("Training col: ", col, " Label col idx: ", col_idx)
                # Next, we train/tune each selected model for that question
