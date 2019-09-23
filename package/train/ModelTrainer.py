"""
QThread for model training
"""
from PyQt5.QtCore import (Qt, pyqtSlot, pyqtSignal,
                          QObject, QThread, QRunnable)

import json
import re
import importlib
import traceback
import inspect
import logging
import os
import pickle
import hashlib
# import threading
import time
from queue import PriorityQueue
from tpot import TPOTClassifier


import pandas as pd
import numpy as np


from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.utils import parallel_backend, register_parallel_backend
from sklearn.preprocessing import FunctionTransformer

from dask.distributed import Client
import joblib
from joblib._parallel_backends import ThreadingBackend, SequentialBackend, LokyBackend
import scipy

from tensorflow.python.keras.preprocessing import sequence, text
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.callbacks import EarlyStopping

import package.utils.training_utils as tu
from package.utils.catutils import CATEncoder, cat_decoder
import package.utils.keras_models as keras_models
import package.utils.embedding_utils as embed_utils
import package.utils.SequenceTransformer as seq_trans

RANDOM_SEED = 1337
TOP_K = 20000
MAX_SEQUENCE_LENGTH = 1500
BASE_MODEL_DIR = "./package/data/base_models"
BASE_TFIDF_DIR = "./package/data/feature_extractors/TfidfVectorizer.json"
INPUT_SHAPE = (0, 0)


class ModelTrainerSignals(QObject):
    training_complete = pyqtSignal(int, bool)
    tuning_complete = pyqtSignal(bool, dict)
    update_progressbar = pyqtSignal(int, bool)
    update_training_logger = pyqtSignal(str)


class ModelTrainer(QRunnable):
    """
    QThread tasked with running all model training/tuning.  
    This could potentially take days to complete.
    """
    # This allows for multi-threading from a thread.  GUI will not freeze and
    # multithreading seems functional.
    # NOTE: some models, e.g. RandomForestClassifier, will not train using this backend.
    # An exception is caught and the log updated if this occurs.
    # register_parallel_backend('threading', ThreadingBackend, make_default=True)
    # parallel_backend('threading')

    def __init__(self, selected_models, version_directory,
                 training_eval_params, training_data,
                 tune_models, n_iter, use_proba=False, train_stacking_algorithm=True, **kwargs
                 ):
        super(ModelTrainer, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.signals = ModelTrainerSignals()

        self.allowed_pipeline_types = [
            'feature_extraction',
            'feature_selection'
        ]
        self.version_directory = version_directory
        # print(self.version_directory)
        self.selected_models = selected_models
        # print(self.selected_models)
        self.training_eval_params = training_eval_params
        # print(json.dumps(self.training_eval_params, indent=2, cls=CATEncoder))
        self.training_data = training_data
        # print(self.training_data.head())
        # print("Tune models?", tune_models)
        self.tune_models = tune_models
        self.n_iter = n_iter
        self.use_proba = use_proba
        self.train_stacking_algorithm = train_stacking_algorithm
        self.kwargs = kwargs
        self.all_predictions_dict = {}
        self.grid_search_time = None
        self.model_checksums = {}

    @pyqtSlot()
    def run(self):
        # Run thru enumeration of columns.  The second argument in enumerate
        # tells python where to begin the idx count.  Here, 1 for our offset
        # training_complete.emit(0, True)
        self._update_log("Beginning ModelTrain run")
        try:
            for col_idx, col in enumerate(self.training_data.columns, 1):
                if col.endswith('_text'):
                    self._update_log(f"Current classification task: {col}")
                    col_label = col.split("_")[0]
                    col_path = os.path.join(self.version_directory, col_label)
                    # Create dict to fill na samples with "unanswered" and score of 0
                    label_col_name = self.training_data.columns[col_idx]
                    fill_dict = pd.DataFrame(
                        data={col: "unanswered", label_col_name: 0}, index=[0])
                    self.training_data.fillna(value=fill_dict, inplace=True)
                    x = self.training_data[col]
                    y = self.training_data[self.training_data.columns[col_idx]].values

                    results = pd.DataFrame(index=self.training_data.index)
                    results['actual'] = y
                    preds = np.empty(y.shape)
                    probs = np.empty(shape=(y.shape[0], len(np.unique(y))))

                    # Initialize sklearn evaluation parameters
                    sk_eval_type = self.training_eval_params['sklearn']['type']
                    sk_eval_value = self.training_eval_params['sklearn']['value']
                    # SKLEARN
                    for model, selected in self.selected_models['sklearn'].items():
                        if selected:
                            try:
                                if self.tune_models:
                                    model_path = os.path.join(".\\package\\data\\default_models\\default",
                                                              model,
                                                              model + '.json')
                                    with open(model_path, 'r') as param_file:
                                        model_params = json.load(
                                            param_file, object_hook=cat_decoder)
                                    #TODO: call pipeline for TPOT using Hyperparameters as key and skip grid_search
                                    pipeline = Pipeline(
                                        self.get_pipeline(model_params['params']))
                                    self._update_log(
                                        f"Begin tuning on {model}")
                                    rscv = self.grid_search(
                                        model, x, y, pipeline, self.n_iter, include_tfidf=True)
                                    if rscv is None:
                                        self._update_log(f"Grid search failed for {model} on task {col}.  Skipping...")
                                        break
                                    else:
                                        preds = rscv.best_estimator_.predict(x)
                                else:
                                    model_path = os.path.join(
                                        col_path, model, model + '.json')
                                    if not os.path.isfile(model_path):
                                        # Get default values
                                        model_path = os.path.join(".\\package\\data\\default_models\\default",
                                                                  model,
                                                                  model + '.json')

                                    # print("model_path", model_path)
                                    with open(model_path, 'r') as param_file:
                                        model_params = json.load(
                                            param_file, object_hook=cat_decoder)
                                    self._update_log(
                                        f"Begin training {model}")

                                    pipeline = Pipeline(
                                        self.get_pipeline(model_params['params']))

                                    if sk_eval_type == 'cv':
                                        skf = StratifiedKFold(n_splits=sk_eval_value,
                                                              random_state=RANDOM_SEED)
                                        try:
                                            for train, test in skf.split(x, y):
                                                with joblib.parallel_backend('dask'):
                                                    preds[test] = pipeline.fit(
                                                        x.iloc[train], y[train]).predict(x.iloc[test])
                                                if self.use_proba and hasattr(pipeline, 'predict_proba'):
                                                    try:
                                                        probs[test] = pipeline.predict_proba(
                                                            x.iloc[test])
                                                    except AttributeError:
                                                        self.logger.debug(
                                                            "{} does not support predict_proba".format(model))
                                                        print(
                                                            model, "does not support predict_proba")
                                                else:
                                                    probs = np.array([])
                                        except Exception:
                                            self.logger.warning(
                                                "{} threw an exception during fit. \
                                                    Possible error with joblib multithreading.".format(model), exc_info=True)
                                            tb = traceback.format_exc()
                                            print(tb)
                                            self._update_log("{} threw an exception during fit. \
                                                    Possible error with joblib multithreading.".format(model))
                                    else:
                                        print("Train called with invalid cv type:", json.dumps(
                                            self.training_eval_params, indent=2, cls=CATEncoder))
                                        return

                                    model_acc = accuracy_score(y, preds)

                                    self._update_log(
                                        f"Training completed on <b>{model}</b>.")
                                    self._update_log(f"Accuracy: {model_acc}")
                                    self._update_log(
                                        f"Training {model} on full dataset")
                                    with joblib.parallel_backend('dask'):
                                        pipeline.fit(x, y)

                                pred_col_name = col_label + '_' + model + '_preds'
                                prob_col_name = col_label + '_' + model + '_probs'
                                results[pred_col_name] = preds.astype(int)
                                # If predicting probabilities and the probability array has values,
                                # use those values for the results.  
                                if self.use_proba and probs.size:
                                    results[prob_col_name] = np.amax(
                                        probs, axis=1)

                                save_path = os.path.join(col_path, model)
                                if not os.path.exists(save_path):
                                    os.makedirs(save_path)
                                save_file = os.path.join(
                                    save_path, model + '.pkl')
                                self._update_log(
                                    f"Saving {model} to : {save_file}")
                                
                                if self.tune_models:
                                    joblib.dump(rscv, save_file, compress=1)
                                    self.model_checksums[model] = hashlib.md5(
                                        open(save_file, 'rb').read()).hexdigest()
                                    self._update_log(
                                        f"{model} checksum: {self.model_checksums[model]}")
                                    self.save_params_to_file(
                                        model, rscv.best_estimator_.get_params(), save_path, rscv.best_score_)
                                else:
                                    joblib.dump(
                                        pipeline, save_file, compress=1)
                                    self.model_checksums[model] = hashlib.md5(
                                        open(save_file, 'rb').read()).hexdigest()
                                    self._update_log(
                                        f"{model} checksum: {self.model_checksums[model]}")
                                    self.save_params_to_file(
                                        model, pipeline.get_params(), save_path, model_acc)
                            except Exception as e:
                                self.logger.error(
                                    "ModelTrainer.run (Sklearn):", exc_info=True)
                                tb = traceback.format_exc()
                                print(tb)
                                self._update_log(tb)
                  # Tensorflow used to reside here

                    if self.train_stacking_algorithm:
                        self.train_stacker(results.drop('actual', axis=1),
                                           results.actual.values,
                                           col_path)

                    # training_complete.emit(0, False)

        except Exception as e:
            self.logger.error("ModelTrainer.run (General):", exc_info=True)
            tb = traceback.format_exc()
            print(tb)
            self._update_log(tb)

    def train_stacker(self, x, y, col_path):
        self._update_log(
            "Training Stacking algorithm (DecisionTreeClassifier)")
        final_preds = np.empty(y.shape)
        encv = DecisionTreeClassifier()
        skf = StratifiedKFold(n_splits=5,
                              random_state=RANDOM_SEED)

        for train, test in skf.split(x, y):
            with joblib.parallel_backend('dask'):
                encv.fit(x.iloc[train], y[train])
            final_preds[test] = encv.predict(x.iloc[test])
        # stack_preds = [1 if x > .5 else 0 for x in np.nditer(final_preds)]
        self._update_log("Stacking training complete")
        stack_acc = accuracy_score(y, final_preds)
        self._update_log(
            f"Stacker score: {stack_acc}")

        save_path = os.path.join(col_path, 'Stacker')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = os.path.join(
            save_path, 'Stacker.pkl')
        self._update_log(f"Saving Stacking algorithm to : {save_file}")
        joblib.dump(encv, save_file, compress=1)
        self.model_checksums['Stacker'] = hashlib.md5(
            open(save_file, 'rb').read()).hexdigest()
        self._update_log(f"Stacking hash: {self.model_checksums['Stacker']}")
        # Save particulars to file
        stacker_info = {
            "column": col_path.split("\\")[-1],
            "version_directory": self.version_directory,
            "last_train_date": time.ctime(time.time()),
            "train_eval_score": stack_acc,
            "model_checksums": self.model_checksums
        }
        stacker_json_save_file = os.path.join(save_path, 'Stacker.json')
        with open(stacker_json_save_file, 'w') as outfile:
            json.dump(stacker_info, outfile, indent=2)

        self._update_log("Run complete")
        self._update_log("********************************************\n")
        # self._update_log(f"<font color='blue'>Idle</font><br>")
        self.signals.training_complete.emit(0, False)

    def get_pipeline(self, param_dict):
        """Builds pipeline steps required for sklearn models.  
            Includes Feature extraction, feature selection, and classifier.
                # Arguments
                    param_dict: dict, dictionary of current model parameter values.
                # Returns
                    pipeline_steps, list: list of steps with intialized classes.
        """
        pipeline_queue = PriorityQueue()
        for args, values in param_dict.items():
            full_class = args.split('.')
            current_module = ".".join(full_class[0:-1])
            current_type = full_class[1]

            if current_type == 'feature_extraction':
                priority = 0
            elif current_type == 'feature_selection':
                priority = 50
            else:
                priority = 100
            inst_module = importlib.import_module(current_module)
            current_class = getattr(inst_module, full_class[-1])
            if values:
                pipeline_queue.put(
                    (priority, (full_class[-1], current_class(**values))))
            else:
                pipeline_queue.put(
                    (priority, (full_class[-1], current_class())))

        pipeline = []
        while not pipeline_queue.empty():
            pipeline.append(pipeline_queue.get()[-1])
        return pipeline

    def grid_search(self,
                    model,
                    x,
                    y,
                    pipeline,
                    n_jobs=-1,
                    n_iter=20,
                    scoring=None,
                    include_tfidf=False,
                    keras_params=None):
        """Performs grid search on selected pipeline.

            # Arguments

                model, string: name of classifier in pipeline
                x, pandas: DataFrame, training data
                y, numpy:array, training labels
                pipeline, sklearn:model_selection.Pipeline, pipeline object containing feature extractors, feature selectors and estimator
                n_iter, int: number of iterations to perform search
                include_tfidf, bool: flag to indicate tfidf is included in the pipeline
                keras_params, dict: parameters necessary for model training outside of the regular hyperparams.  e.g. input_shape, num_classes, num_features
        """
        try:
            start_time = time.time()
            filepath = os.path.join(BASE_MODEL_DIR, model + '.json')
            with open(filepath, 'r') as f:
                # print("Loading model:", filepath)
                model_data = json.load(f, object_hook=cat_decoder)
            print("model_data:")
            print(json.dumps(model_data, indent=2))
            grid_params = {}
            default_params = model_data[model]

            for param_types, types in default_params.items():
                for t, params in types.items():
                    if params['tunable']:
                        param_name = model + '__' + t
                        if params['type'] == 'dropdown':
                            param_options = list(params['options'].values())
                        elif params['type'] == 'double':
                            param_options = scipy.stats.expon(
                                scale=params['step_size'])
                        elif params['type'] == 'int':
                            param_options = scipy.stats.randint(
                                params['min'], params['max'] + 1)
                        elif params['type'] == 'range':
                            param_options = [(1, 1), (1, 2), (1, 3), (1, 4)]
                        grid_params.update({param_name: param_options})
                    else:
                        continue

            if include_tfidf:
                with open(BASE_TFIDF_DIR, 'r') as f:
                    model_data = json.load(f, object_hook=cat_decoder)
                model_class = model_data['model_class']
                default_params = model_data[model_class]

                for param_types, types in default_params.items():
                    for t, params in types.items():
                        if params['tunable']:
                            param_name = model_class + '__' + t
                            if params['type'] == 'dropdown':
                                param_options = list(
                                    params['options'].values())
                            elif params['type'] == 'double':
                                param_options = scipy.stats.expon(
                                    scale=params['step_size'])
                            elif params['type'] == 'int':
                                param_options = scipy.stats.randint(
                                    params['min'], params['max'] + 1)
                            elif params['type'] == 'range':
                                param_options = [
                                    (1, 1), (1, 2), (1, 3), (1, 4)]
                            else:
                                param_options = None
                            grid_params.update({param_name: param_options})
                        else:
                            continue
            # Remnant from __TENSORFLOW work.  
            # if keras_params:
            #     updated_key_dict = {f'{model}__{k}': 
            #         [v] for k, v in keras_params.items()}
            #     grid_params.update(updated_key_dict)
            # FIXME: I'm popping this by idx.  This is a serious no-no.
            # find a better way to remove feature selection from pipeline.
            if 'SelectPercentile' in pipeline.named_steps:
                pipeline.steps.pop(1)

            self._update_log(f"Beginning RandomizedSearchCV on {model}...")
            # print("Params: ", grid_params)
            # print("Pipeline:", [name for name, _ in pipeline.steps])
            rscv = RandomizedSearchCV(pipeline,
                                      grid_params,
                                      n_jobs=n_jobs,
                                      cv=3,
                                      n_iter=n_iter,
                                      pre_dispatch='3*n_jobs',
                                      verbose=10,
                                      refit=True)
            with joblib.parallel_backend('dask'):
                rscv.fit(x, y)
            self.grid_search_time = time.time() - start_time
            self._update_log(
                f"RandomizedSearchCV on {model} completed in {self.grid_search_time}")
            self._update_log(f"Best score for {model}: {rscv.best_score_}")
            return rscv

        except FileNotFoundError as fnfe:
            self.logger.debug(
                "ModelTrainer.grid_search {} not found".format(filepath))
        except Exception as e:
            self.logger.error(
                "ModelTrainer.grid_search {}:".format(model), exc_info=True)
            tb = traceback.format_exc()
            print(tb)
            self._update_log(tb)

    def save_params_to_file(self, model, best_params, model_param_path, best_score):
        try:
            model_path = os.path.join(model_param_path, model, model + '.json')
            if not os.path.isfile(model_path):
                # Get default values
                model_path = os.path.join(".\\package\\data\\default_models\\default",
                                          model,
                                          model + '.json')
            with open(model_path, 'r') as param_file:
                model_params = json.load(param_file)

            model_params['meta'] = {
                "training_meta": {
                    "last_train_date": time.ctime(time.time()),
                    "train_eval_score": best_score,
                    "checksum": self.model_checksums[model]
                },

            }
            if self.tune_models:
                model_params['meta']["tuning_meta"] = {
                    "last_tune_date": time.ctime(time.time()),
                    "n_iter": self.n_iter,
                    "tuning_duration": self.grid_search_time,
                    "tune_eval_score": best_score
                }

            # Update model params to the best
            for param_type, parameters in model_params['params'].items():
                param_key = param_type.split('.')[-1]
                for k, v in best_params.items():
                    best_param_key = k.split('__')[-1]
                    if k.startswith(param_key) and best_param_key in parameters.keys():
                        parameters[best_param_key] = v

            # print(
            #     f"***** Saving {model} best_params to {model_param_path}....")
            with open(os.path.join(model_param_path, model + '.json'), 'w') as outfile:
                json.dump(model_params, outfile, indent=2, cls=CATEncoder)

        except FileNotFoundError as fnfe:
            self.logger.debug(
                "ModelTrainer.save_params_to_file {} not found".format(model_path))
        except Exception as e:
            self.logger.error(
                "ModelTrainer.save_params_to_file {}:".format(model), exc_info=True)
            tb = traceback.format_exc()
            print(tb)

    def _generate_best_param_dict(self, model_param_keys, best_params):
        try:
            result_dict = {el: {} for el in model_param_keys}
            for param_type in model_param_keys:
                key = param_type.split('.')[-1]
                result_dict[key] = {
                    k: v for k, v in best_params.items() if k.startswith(key.split('.')[-1])}

            return result_dict
        except Exception as e:
            self.logger.error(
                "ModelTrainer._generate_best_param_dict {}:".format(e), exc_info=True)
            tb = traceback.format_exc()
            print(tb)

    def _update_log(self, msg):
        outbound = f"{time.ctime(time.time())} - {msg}<br>"
        self.signals.update_training_logger.emit(outbound)

    @pyqtSlot()
    def stop_thread(self):
        self._update_log("Stopping ModelTrainer...")
        # TODO: Add funtionality to stop the thread
        self.__abort = True
        self.quit()
        self.wait()
