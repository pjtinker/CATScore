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
from sklearn.preprocessing import FunctionTransformer

import joblib
from joblib._parallel_backends import ThreadingBackend

import scipy

from tensorflow.python.keras.preprocessing import sequence, text
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.callbacks import EarlyStopping

import package.utils.training_utils as tu
import package.utils.keras_models as keras_models
import package.utils.embedding_utils as embed_utils
import package.utils.SequenceTransformer as seq_trans

RANDOM_SEED = 1337
TOP_K = 20000
MAX_SEQUENCE_LENGTH = 1500
BASE_MODEL_DIR = "./package/data/base_models"
BASE_TFIDF_DIR = "./package/data/feature_extractors/TfidfVectorizer.json"
INPUT_SHAPE = (0, 0)

class ModelTrainer(QRunnable):
    """
    QThread tasked with running all model training/tuning.  
    This could potentially take days to complete.
    """
    # training_complete = pyqtSignal(int, bool)
    # update_progressbar = pyqtSignal(int, bool)

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
        self.n_iter = n_iter
        self.kwargs = kwargs
        self.all_predictions_dict = {}
    
    def run(self):
        # Run thru enumeration of columns.  The second argument in enumerate
        # tells python where to begin the idx count.  Here, 1 for our offset
        # training_complete.emit(0, True)
        try:
            for col_idx, col in enumerate(self.training_data.columns, 1): 
                if col.endswith('_Text'):
                    print("Training col: ", col, " Label col idx: ", col_idx)
                    print("training data head:")
                    print(self.training_data[col].head())
                    print("training data label head:")
                    print(self.training_data[self.training_data.columns[col_idx]].head())

                    col_label = col.split("_")[0]
                    col_path = os.path.join(self.version_directory, col_label)
                    print("col_path:", col_path)

                    # self.all_predictions_dict[col_label] = pd.DataFrame()
                    results = pd.DataFrame()
                    
                    x = self.training_data[col]
                    y = self.training_data[self.training_data.columns[col_idx]].values
                    preds = np.empty(y.shape)
                    probs = np.empty(shape=(y.shape[0], len(np.unique(y))))

                    # Initialize all evaluation parameters
                    sk_eval_type = self.training_eval_params['sklearn']['type']
                    sk_eval_value = self.training_eval_params['sklearn']['value']

                    # Init tensorflow eval params
                    patience = self.training_eval_params['tensorflow']['patience']
                    if patience > 0:
                        callbacks = [EarlyStopping(monitor='val_loss', 
                                                    patience=patience)]
                    else:
                        callbacks = None
                        
                    validation_split = self.training_eval_params['tensorflow']['validation_split']

                    use_pretrained_embedding = self.training_eval_params['tensorflow']['use_pretrained_embedding']

                    if use_pretrained_embedding:
                        embedding_type = self.training_eval_params['tensorflow']['embedding_type']
                        embedding_dim = self.training_eval_params['tensorflow']['embedding_dim']
                    else:
                        embedding_type = None
                        embedding_dim = None

                    eu = embed_utils.EmbeddingUtils(embedding_type, embedding_dim)
                    is_embedding_trainable = self.training_eval_params['tensorflow']['is_embedding_trainable']
                    embedding_dim = self.training_eval_params['tensorflow']['embedding_dim']

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
                                    rscv = self.grid_search(model, x, y, pipeline, self.n_iter, True)
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
                                    
                                    if sk_eval_type == 'cv':
                                        skf = StratifiedKFold(n_splits=sk_eval_value,
                                                                random_state=RANDOM_SEED)

                                        for train, test in skf.split(x, y):
                                            preds[test] = pipeline.fit(x.iloc[train], y[train]).predict(x.iloc[test])
                                            try:
                                                probs[test] = pipeline.predict_proba(x.iloc[test])
                                            except AttributeError:
                                                self.logger.info("{} does not support predict_proba".format(model))
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
                        tokenizer = text.Tokenizer(num_words=TOP_K)
                        tokenizer.fit_on_texts([str(word) for word in x])
                        
                        sequence_transformer = seq_trans.SequenceTransformer(tokenizer, TOP_K)
                        num_features = min(len(tokenizer.word_index) + 1, TOP_K)
                        num_classes = len(np.unique(y))
                        
                        if self.training_eval_params['tensorflow']['use_pretrained_embedding']:
                            use_pretrained_embedding = True
                            eu.generate_embedding_matrix(tokenizer.word_index)
                        else:
                            use_pretrained_embedding = False

                        for model, truth in self.selected_models['tensorflow'].items():
                            print("Current model from os.listdir(col_path)", model)
                            if truth:
                                try:
                                    if self.tune_models:
                                        param_dict = dict(  
                                                            input_shape=INPUT_SHAPE,
                                                            num_classes=num_classes,
                                                            num_features=num_features,
                                                            embedding_dim=embedding_dim,
                                                            use_pretrained_embedding=use_pretrained_embedding,
                                                            embedding_matrix=eu.get_embedding_matrix()
                                                        )

                                        keras_model = KerasClassifier(build_fn=keras_models.sepcnn_model)
                                        pipeline = Pipeline(steps=[('SequenceTransformer', sequence_transformer),
                                                                   (model, keras_model)] )

                                        rscv = self.grid_search(model, x, y, pipeline, self.n_iter, False, param_dict)
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

                                        param_dict = dict(input_shape=INPUT_SHAPE,
                                                          num_classes=num_classes,
                                                          num_features=num_features,
                                                          validation_split=validation_split,
                                                          callbacks=callbacks,
                                                          embedding_dim=embedding_dim,
                                                          use_pretrained_embedding=use_pretrained_embedding,
                                                          is_embedding_trainable=is_embedding_trainable,
                                                          embedding_matrix=eu.get_embedding_matrix()
                                                        )
                                        param_dict.update(model_params['params'][
                                            ".".join((model_params['model_module'], model_params['model_class']))
                                        ])
                                        print(model + " params:", param_dict)
                                        keras_model = KerasClassifier(build_fn=keras_models.sepcnn_model, **param_dict)
                                        kc = Pipeline(steps=[('SequenceTransformer', sequence_transformer ),
                                                                   (model, keras_model) ] )

                                        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                                            test_size = validation_split,
                                                                                            shuffle=True,
                                                                                            stratify=y)
                                        kc.fit(x_train, y_train)
                                        tf_preds = kc.predict(x_test)
                                        print("Accuracy: ", accuracy_score(y_test, tf_preds))
                                        history = kc.fit(x, y)

                                    save_path = os.path.join(col_path, model)
                                    if not os.path.exists(save_path):
                                        os.makedirs(save_path)
 
                                    save_file = os.path.join(save_path, model + '.h5')
                                    print("Saving model to :", save_path)
                                    if self.tune_models:
                                        best_params = rscv.best_params_
                                        kc = rscv.best_estimator_

                                        # param_file = os.path.join(model + '.json')
                                        print(f"Saving {model} params to {param_file}...")
                                        # If we used pretrained embeddings, they've been saved
                                        # as a parameter.  Delete them for space and simplicity.  

                                        best_params.pop(model + '__embedding_matrix', None)
                                        #TODO: Return best params to SelectModelWidget
                                    # Keras model must be saved separately and removed from pipeline
                                    kc.named_steps[model].model.save(save_file)
                                    kc.named_steps[model].model = None
                                    # Save pipeline with Keras model deleted
                                    pipeline_save_file = os.path.join(save_path, model + '_pipeline.pkl')
                                    print(f"Saving tuned {model} model to {pipeline_save_file}...")
                                    joblib.dump(kc, pipeline_save_file, compress=1)
                                    
                                    del kc
                                except Exception as e:
                                    self.logger.error("ModelTrainer.run (Tensorflow):", exc_info=True)
                                    tb = traceback.format_exc()
                                    print(tb)
        # training_complete.emit(0, False)  
        except Exception as e:
            self.logger.error("ModelTrainer.run (General):", exc_info=True)
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

    def grid_search(self, model, x, y, pipeline, n_iter=2, include_tfidf=False, keras_params=None):
        """Performs grid search on selected pipeline.
            # Arguments
                model: string, name of classifier in pipeline
                x: pandas.DataFrame, training data
                y: numpy.array, training labels
                pipeline: sklearn.model_selection.Pipeline, pipeline object containing feature extractors, feature selectors and estimator
                n_iter: int, number of iterations to perform search
                include_tfidf: bool, flag to indicate tfidf is included in the pipeline
                keras_params: dict, parameters necessary for model training outside of the regular hyperparams.  e.g. input_shape, num_classes, num_features
        """
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
            if keras_params:
                updated_key_dict = {f'{model}__{k}': [v] for k, v in keras_params.items()}
                grid_params.update(updated_key_dict)

            if 'SelectKBest' in pipeline.named_steps:
                grid_params['SelectKBest__k'] = ['all']

            print(f"Beginning RandomizedSearchCV on {model}...")
            print("Params: ", grid_params)
            print("Pipeline:", [name for name, _ in pipeline.steps])
            rscv = RandomizedSearchCV(pipeline, grid_params, n_jobs=-1, cv=3, n_iter=n_iter, refit=True)
            rscv.fit(x, y)
                
            print("Best score:")
            print(rscv.best_score_)
            print("Best params:")
            print(rscv.best_params_)
            return rscv

        except FileNotFoundError as fnfe:
            self.logger.debug("ModelTrainer.grid_search {} not found".format(filepath))
        except Exception as e:
            self.logger.error("ModelTrainer.grid_search {}:".format(model), exc_info=True)
            tb = traceback.format_exc()
            print(tb)

    
