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
from sklearn.model_selection import cross_val_score

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
        self.allowed_pipeline_types = [
            'feature_extraction',
            'feature_selection'
        ]
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
        # tells python where to begin the idx count.  Here, 1 for our offset
        for col_idx, col in enumerate(self.training_data.columns, 1): 
            if col.endswith('_Text'):
                print("Training col: ", col, " Label col idx: ", col_idx)
                print("training data head:")
                print(self.training_data[col].head())
                print("training data label head:")
                print(self.training_data[self.training_data.columns[col_idx]].head())
                # sklearn cross_val_score should be used for evaluation, not 
                # calling an eval method using cross_val_predict.
                # I will use cross_val_predict for confusion matrix only.
                col_label = col.split("_")[0]
                col_path = os.path.join(self.version_directory, col_label)
                print("col_path:", col_path)
                for models, truth in self.selected_models['sklearn'].items():
                    # for model, truth in self.selected_models['sklearn'].items():
                    print("Current model from os.listdir(col_path)", models)
                    if truth:
                        # if truth:
                        try:
                            model_path = os.path.join(col_path, models, models + '.json')
                            if not os.path.isfile(model_path):
                                # with open(model_path, 'r') as param_file:
                                #     model_params = json.load(param_file)
                                # print("Loaded params from: ", model_path)
                                # print(json.dumps(model_params, indent=2))
                                model_path = os.path.join(".\\package\\data\\default_models\\default",
                                                            models,
                                                            models + '.json')
                            # else:
                            print("model_path", model_path)
                            with open(model_path, 'r') as param_file:
                                model_params = json.load(param_file)

                            # pipeline = self.get_pipeline(model_params['params'])
                            learners = self.get_pipeline(model_params['params'])

                            x = self.training_data[col]
                            y = self.training_data[self.training_data.columns[col_idx]]

                            x_vecs = learners[0].fit_transform(x)
                            learners[1].fit(x_vecs, y)
                            x_selected = learners[1].transform(x_vecs).astype('float32')
                            # scores = cross_val_score(pipeline, 
                            scores = cross_val_score(learners[2],
                                                    #  self.training_data[col],
                                                    #  self.training_data[self.training_data.columns[col_idx]],
                                                     x_selected,
                                                     y,
                                                     cv=3,
                                                     scoring='accuracy')

                            print(scores.mean())
                        except Exception as e:
                            print(e)
                            tb = traceback.format_exc()
                            print(tb)


    def get_pipeline(self, param_dict):
        pipeline_steps = [None] * len(param_dict)
        for args, values in param_dict.items():
            full_class = args.split('.')
            current_module = ".".join(full_class[0:-1])
            current_type = full_class[1]
            if current_type == 'feature_extraction': 
                idx = 0
            elif current_type == 'feature_selection':
                idx = 1
            else:   
                idx = 2
            print("Loading module ", current_module, "Class: ", full_class[-1])
            inst_module = importlib.import_module(current_module)
            current_class = getattr(inst_module, full_class[-1])
            if values:
                if 'ngram_range' in values.keys():
                    values['ngram_range'] = (1, values['ngram_range'])
                if 'score_func' in values.keys():
                    fs_module = importlib.import_module("sklearn.feature_selection")
                    fs_class = getattr(fs_module, values['score_func'])
                    values['score_func'] = fs_class
                step = current_class(**values)
            else:
                step = current_class()
            print("Params for ", args)
            print(step.get_params())
            pipeline_steps[idx] = step


        return pipeline_steps
