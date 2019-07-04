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
import pickle 

import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.externals import joblib

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
        self.logger = logging.getLogger(__name__)
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

        self.all_predictions_dict = {}
    
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

                # self.all_predictions_dict[col_label] = pd.DataFrame()
                results = pd.DataFrame()

                for model, truth in self.selected_models['sklearn'].items():
                    # for model, truth in self.selected_models['sklearn'].items():
                    print("Current model from os.listdir(col_path)", model)
                    if truth:
                        try:
                            model_path = os.path.join(col_path, model, model + '.json')
                            if not os.path.isfile(model_path):
                                # Get default values
                                model_path = os.path.join(".\\package\\data\\default_models\\default",
                                                            model,
                                                            model + '.json')

                            print("model_path", model_path)
                            with open(model_path, 'r') as param_file:
                                model_params = json.load(param_file)

                            # pipeline = make_pipeline(*self.get_pipeline(model_params['params']))
                            pipeline = Pipeline(self.get_pipeline(model_params['params']))
                            # learners = self.get_pipeline(model_params['params'])

                            x = self.training_data[col]
                            y = self.training_data[self.training_data.columns[col_idx]].values
                            preds = np.empty(y.shape)
                            probs = np.empty(shape=(y.shape[0], len(np.unique(y))))

                            if self.training_eval_params['sklearn']['type'] == 'cv':
                                skf = StratifiedKFold(n_splits=self.training_eval_params['sklearn']['value'])

                                for train, test in skf.split(x, y):
                                    preds[test] = pipeline.fit(x.iloc[train], y[train]).predict(x.iloc[test])
                                    try:
                                        probs[test] = pipeline.predict_proba(x.iloc[test])
                                    except AttributeError:
                                        self.logger.debug("{} does not support predict_proba".format(model))
                                        print(model, "does not support predict_proba")
                                        
                            else:
                                print("Training_eval_params:", json.dumps(self.training_eval_params))
                            pred_col_name = col_label + '_' + model + '_preds'
                            prob_col_name = col_label + '_' + model + '_probs'
                            results[pred_col_name] = preds.astype(int)
                            if probs.size:
                                results[prob_col_name] = np.amax(probs, axis=1)
                            # results[col_label + ]
                            # print(results.head())
                            # x_vecs = learners[0].fit_transform(x)
                            # learners[1].fit(x_vecs, y)
                            # x_selected = learners[1].transform(x_vecs).astype('float32')
                            # scores = cross_val_score(pipeline, 
                            # scores = cross_val_score(learners[2],
                                                    #  self.training_data[col],
                                                    #  self.training_data[self.training_data.columns[col_idx]],
                                                    #  x_selected,
                                                    #  x,
                                                    #  y,
                                                    #  cv=3,
                                                    #  scoring='accuracy')

                            print("Accuracy: ", accuracy_score(y, preds))
                            
                            pipeline.fit(x,y)
                            print(pipeline.named_steps)
                            clf = pipeline.named_steps[model]

                            save_path = os.path.join(col_path, model)
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            save_file = os.path.join(save_path, model + '.pkl')
                            print("Saving model to :", save_path)
                            pickle.dump(clf, open(save_file, 'wb'))
                        except Exception as e:
                            self.logger.error("Exception occured in ModelTrainer", exc_info=True)
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
                step = (full_class[-1], current_class(**values))
            else:
                step = (full_class[-1], current_class())
            # print("Params for ", args)
            # print(step.get_params())
            pipeline_steps[idx] = step


        return pipeline_steps
