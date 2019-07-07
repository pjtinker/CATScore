"""
QThread for model training
"""
from PyQt5.QtCore import (Qt, pyqtSlot, pyqtSignal, QObject, QThread, QRunnable)

import json
import re
import importlib 
import traceback
import inspect
import logging
import os
import pickle 
import threading

import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.utils import parallel_backend, register_parallel_backend

import joblib
from joblib._parallel_backends import ThreadingBackend

import scipy

from tensorflow.python.keras.preprocessing import sequence, text
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.callbacks import EarlyStopping

import package.utils.training_utils as tu
import package.utils.keras_models as keras_models
import package.utils.embedding_utils as embed_utils

RANDOM_SEED = 1337
TOP_K = 20000
MAX_SEQUENCE_LENGTH = 1000
BASE_MODEL_DIR = "./package/data/base_models"
BASE_TFIDF_DIR = "./package/data/feature_extractors/TfidfVectorizer.json"

class ModelTrainer(QRunnable):
    """
    QThread tasked with running all model training/tuning.  
    This could potentially take days to complete.
    """
    training_complete = pyqtSignal(pd.DataFrame)
    register_parallel_backend('threading', ThreadingBackend, make_default=True)

    def __init__(self, selected_models, version_directory, 
                 training_eval_params, training_data, 
                 tune_models, n_iter, **kwargs
                ):
        super(ModelTrainer, self).__init__()
        # threading.current_thread().name == 'MainThread'
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
        print("Tune models?", tune_models)
        self.tune_models = tune_models
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
                
                x = self.training_data[col]
                y = self.training_data[self.training_data.columns[col_idx]].values
                preds = np.empty(y.shape)
                probs = np.empty(shape=(y.shape[0], len(np.unique(y))))


                for model, truth in self.selected_models['sklearn'].items():
                    print("Current model from os.listdir(col_path)", model)
                    if truth:
                        if model == 'TPOTClassifier':
                            print("TPOTClassifier's turn.  Doing special thing...")
                            continue
                        try:
                            if self.tune_models:
                                model_path = os.path.join(".\\package\\data\\default_models\\default",
                                                                model,
                                                                model + '.json')

                                print("model_path", model_path)
                                with open(model_path, 'r') as param_file:
                                    model_params = json.load(param_file)

                                pipeline = Pipeline(self.get_pipeline(model_params['params']))
                                rscv = self.grid_search(model, x, y, pipeline, 20, True)
                            else:                               
                                model_path = os.path.join(col_path, model, model + '.json')
                                if not os.path.isfile(model_path):
                                    # Get default values
                                    model_path = os.path.join(".\\package\\data\\default_models\\default",
                                                                model,
                                                                model + '.json')

                                print("model_path", model_path)
                                with open(model_path, 'r') as param_file:
                                    model_params = json.load(param_file)

                                pipeline = Pipeline(self.get_pipeline(model_params['params']))
                                
                                if self.training_eval_params['sklearn']['type'] == 'cv':
                                    skf = StratifiedKFold(n_splits=self.training_eval_params['sklearn']['value'],
                                                            random_state=RANDOM_SEED)

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

                                print("Accuracy: ", accuracy_score(y, preds))
                                
                                pipeline.fit(x,y)
                                print(pipeline.named_steps)
                                # clf = pipeline.steps[-1:]

                            save_path = os.path.join(col_path, model)
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            save_file = os.path.join(save_path, model + '.pkl')
                            print("Saving model to :", save_path)
                            if self.tune_models:
                                joblib.dump(rscv, save_file, compress=1)
                            else:
                                joblib.dump(pipeline, save_file, compress=1)
                        except Exception as e:
                            self.logger.error("ModelTrainer.run (Sklearn):", exc_info=True)
                            tb = traceback.format_exc()
                            print(tb)
                # TENSORFLOW BEGINS
                if (1 in self.selected_models['tensorflow'].values()):
                    print("Tokenizing input text for Tensorflow models...")
                    tokenizer = text.Tokenizer(num_words=TOP_K)
                    tokenizer.fit_on_texts([str(word) for word in x])
                    xdata = tokenizer.texts_to_sequences([str(word) for word in x])

                    max_len = len(max(xdata, key=len))
                    if max_len > MAX_SEQUENCE_LENGTH:
                        max_len = MAX_SEQUENCE_LENGTH

                    xdata = sequence.pad_sequences(xdata, maxlen=max_len)

                    num_classes = len(np.unique(y))
                    num_features = min(len(tokenizer.word_index) + 1, TOP_K)

                    eu = embed_utils.EmbeddingUtils()
                    if self.training_eval_params['tensorflow']['embedding_trainable']:
                        use_pretrained_embedding = True
                        eu.generate_embedding_matrix(tokenizer.word_index)
                    else:
                        use_pretrained_embedding = False

                    for model, truth in self.selected_models['tensorflow'].items():
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
                                print(model_params)
                                patience = self.training_eval_params['tensorflow']['patience']
                                if patience > 0:
                                    callbacks = [EarlyStopping(monitor='val_loss', 
                                                                patience=4)]
                                else:
                                    callbacks = None

                                param_dict = dict(input_shape=xdata.shape[1:],
                                                    num_classes=num_classes,
                                                    num_features=num_features,
                                                    epochs=100,
                                                    validation_split=0.2,
                                                    batch_size=64,
                                                    callbacks=callbacks,
                                                    use_pretrained_embedding=use_pretrained_embedding,
                                                    embedding_matrix=eu.get_em())

                                model = KerasClassifier(build_fn=keras_models.sepcnn_model, **param_dict)
                                x_train, x_test, y_train, y_test = train_test_split(xdata, y,
                                                                                    test_size = self.training_eval_params['tensorflow']['validation_split'],
                                                                                    shuffle=True,
                                                                                    stratify=y)
                                history = model.fit(x_train, y_train)
                                tf_preds = model.predict(x_test)
                                print("Accuracy: ", accuracy_score(y_test, tf_preds))


                            except Exception as e:
                                self.logger.error("ModelTrainer.run (Tensorflow):", exc_info=True)
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
            pipeline_steps[idx] = step


        return pipeline_steps

    def grid_search(self, model, x, y, pipeline, n_iter=10, include_tfidf=False):
        try:
            filepath = os.path.join(BASE_MODEL_DIR, model + '.json')
            with open(filepath, 'r') as f:
                print("Loading model:", filepath)
                model_data = json.load(f)
            grid_params = {}
            default_params = model_data[model]
            
            for param_types, types in default_params.items():
                for t, params in types.items():
                    if params['tunable']:
                        param_name = model + '__' + t
                        if params['type'] == 'dropdown':
                            param_options = list(params['options'].values())
                        elif params['type'] == 'double':
                            param_options = scipy.stats.expon(scale=params['step_size'])
                        elif params['type'] == 'int':
                            param_options = scipy.stats.randint(params['min'], params['max'] + 1)
                        elif params['type'] == 'range':
                            param_options = [(1,1), (1,2), (1,3), (1,4)]
                        grid_params.update({param_name : param_options})
                    else:
                        continue

            if include_tfidf:
                with open(BASE_TFIDF_DIR, 'r') as f:
                    print("Loading model:", BASE_TFIDF_DIR)
                    model_data = json.load(f)
                model_class = model_data['model_class']
                default_params = model_data[model_class]
                
                for param_types, types in default_params.items():
                    for t, params in types.items():
                        if params['tunable']:
                            param_name = model_class + '__' + t
                            if params['type'] == 'dropdown':
                                param_options = list(params['options'].values())
                            elif params['type'] == 'double':
                                param_options = scipy.stats.expon(scale=params['step_size'])
                            elif params['type'] == 'int':
                                param_options = scipy.stats.randint(params['min'], params['max'] + 1)
                            elif params['type'] == 'range':
                                param_options = [(1,1), (1,2), (1,3), (1,4)]
                            grid_params.update({param_name : param_options})
                        else:
                            continue
            
            if 'SelectKBest' in pipeline.named_steps:
                grid_params['SelectKBest__k'] = ['all']

            print("Beginning RandomizedSearchCV...")
            print("Params: ", grid_params)
            print("Pipeline:", [name for name, _ in pipeline.steps])
            rscv = RandomizedSearchCV(pipeline, grid_params, n_jobs=-1, cv=3, n_iter=n_iter, refit=True)
            rscv.fit(x, y)
                
            print("Best score:")
            print(rscv.best_score_)
            return rscv.best_estimator_

        except FileNotFoundError as fnfe:
            self.logger.debug("ModelTrainer.grid_search {} not found".format(filepath))
        except Exception as e:
            self.logger.error("ModelTrainer.grid_search {}:".format(model), exc_info=True)
            tb = traceback.format_exc()
            print(tb)

    