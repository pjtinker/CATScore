'''
QThread for model training
'''
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


import pandas as pd
import numpy as np

from scipy.stats import uniform

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, balanced_accuracy_score, f1_score, classification_report, cohen_kappa_score
from sklearn.utils import parallel_backend, register_parallel_backend
from sklearn.preprocessing import FunctionTransformer

from xgboost.sklearn import XGBClassifier
from tpot import TPOTClassifier
import joblib
from joblib._parallel_backends import ThreadingBackend, SequentialBackend, LokyBackend

import scipy
# from tensorflow.python.keras.preprocessing import sequence, text
# from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
# from tensorflow.python.keras.callbacks import EarlyStopping

import package.utils.training_utils as tu
from package.utils.catutils import CATEncoder, cat_decoder, exceptionWarning
from package.utils.config import CONFIG
import package.utils.embedding_utils as embed_utils
# import package.utils.keras_models as keras_models
# import package.utils.SequenceTransformer as seq_trans


RANDOM_SEED = 1337
TOP_K = 20000
MAX_SEQUENCE_LENGTH = 1500
BASE_MODEL_DIR = './package/data/base_models'
BASE_TFIDF_DIR = './package/data/feature_extractors/TfidfVectorizer.json'
INPUT_SHAPE = (0, 0)

TAG_DELIMITER = CONFIG.get('VARIABLES', 'TagDelimiter')
PRED_LABEL_SUFFIX = CONFIG.get('VARIABLES', 'PredictedLabelSuffix')
PROB_LABEL_SUFFIX = CONFIG.get('VARIABLES', 'ProbabilityLabelSuffix')
TRUTH_LABEL_SUFFIX = CONFIG.get('VARIABLES', 'TruthLabelSuffix')
STACKER_LABEL_SUFFIX = CONFIG.get('VARIABLES', 'StackerLabelSuffix')
DISAGREEMENT_THRESHOLD = CONFIG.getfloat('VARIABLES', 'DisagreementThreshold')
BAMBOOZLED_THRESHOLD = CONFIG.getint('VARIABLES', 'BamboozledThreshold')

class ModelTrainerSignals(QObject):
    training_complete = pyqtSignal(pd.DataFrame)
    tuning_complete = pyqtSignal(bool, dict)
    update_progressbar = pyqtSignal(int, bool)
    update_training_logger = pyqtSignal(str, bool, bool)


class ModelTrainer(QRunnable):
    '''
    QThread tasked with running all model training/tuning.  
    This could potentially take days to complete.
    '''
    # Setting parallel_backend to threading allows for multi-threading from a thread.  GUI will not freeze and
    # multithreading seems functional.
    # However, program now uses dask for the backend.  This code is left in for posterity
    # ! NOTE: some models, e.g. RandomForestClassifier, will not train using any backend attempted when n_jobs > 1.
    # ! This is regardless of using dask or joblib backend.  RandomForestClassifier will fail with n_jobs > 1
    # An exception is caught and the log updated if this occurs.
    register_parallel_backend('threading', ThreadingBackend, make_default=True)
    parallel_backend('threading')

    def __init__(
        self,
        selected_models,
        version_directory,
        training_eval_params,
        training_data,
        tune_models,
        tuning_params,
        use_proba=False,
        train_stacking_algorithm=True,
        **kwargs
    ):
        super(ModelTrainer, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.signals = ModelTrainerSignals()

        self.allowed_pipeline_types = [
            'feature_extraction',
            'feature_selection'
        ]
        self.version_directory = version_directory
        self.selected_models = selected_models
        self.training_eval_params = training_eval_params
        self.training_data = training_data
        self.tune_models = tune_models
        self.tuning_params = tuning_params
        self.use_proba = use_proba
        self.train_stacking_algorithm = train_stacking_algorithm
        self.kwargs = kwargs
        self.all_predictions_df = pd.DataFrame(index=self.training_data.index)
        self.grid_search_time = None
        self.model_checksums = {}
        self._is_running = True
        self.tag_suffix = CONFIG.get(
            'VARIABLES', 'TagDelimiter') + CONFIG.get('VARIABLES', 'TagDataColumnSuffix')

    @pyqtSlot()
    def run(self):
        self._update_log('Beginning ModelTrain run')
        # * Run thru enumeration of columns.  The second argument in enumerate
        # * tells python where to begin the idx count.  Here, 1 for our offset
        try:
            for col_idx, col in enumerate(self.training_data.columns, 1):
                if col.endswith(self.tag_suffix):
                    self._update_log(
                        f'Current classification task: {col}', False)
                    col_label = col.split(CONFIG.get(
                        'VARIABLES', 'TagDelimiter'))[0]
                    col_path = os.path.join(self.version_directory, col_label)
                    # * FInd and drop any samples missing an index
                    missing_idx_count = self.training_data.index.isna().sum()
                    if(missing_idx_count > 0):
                        self._update_log(f"<b>Found {missing_idx_count} samples missing a value for index </b> \
                                        (index_col = {CONFIG.get('VARIABLES', 'IndexColumn')}).  Removing those samples...")
                        valid_indexes = self.training_data.index.dropna()
                        self.training_data = self.training_data[self.training_data.index.isin(
                            valid_indexes)]
                        self._update_log(
                            f'Shape of dataset after removal: {self.training_data.shape}')
                    # * Create dict to fill na samples with 'unanswered' and score of 0
                    label_col_name = self.training_data.columns[col_idx]
                    fill_dict = pd.DataFrame(
                        data={col: 'unanswered', label_col_name: 0}, index=[0])
                    self.training_data.fillna(value=0, inplace=True, axis=1)
                    x = self.training_data[col].copy()
                    y = self.training_data[self.training_data.columns[col_idx]].copy(
                    ).values

                    results = pd.DataFrame(index=self.training_data.index)
                    results[TRUTH_LABEL_SUFFIX] = y
                    preds = np.empty(y.shape)
                    probs = np.empty(shape=(y.shape[0], len(np.unique(y))))

                    # * Initialize sklearn evaluation parameters
                    sk_eval_type = self.training_eval_params['sklearn']['type']
                    sk_eval_value = self.training_eval_params['sklearn']['value']
                    # * SKLEARN
                    for model, selected in self.selected_models['sklearn'].items():
                        if self._is_running == False:
                            self.signals.training_complete.emit(pd.DataFrame())
                            break
                        if selected:
                            try:
                                if self.tune_models:
                                    self._tune_model(x, y, model, col_path)
                                model_params = self.get_params_from_file(
                                    model, col_path)
                                self._update_log(
                                    f'Begin training {model}')
                                pipeline = Pipeline(
                                    self.get_pipeline(model_params['params']))
                                try:
                                    if sk_eval_type == 'cv':
                                        skf = StratifiedKFold(n_splits=sk_eval_value,
                                                              random_state=RANDOM_SEED,
                                                              shuffle=True)
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
                                                        '{} does not support predict_proba'.format(model))
                                                    print(
                                                        model, 'does not support predict_proba')
                                            else:
                                                probs = np.array([])
                                    elif sk_eval_type == 'test_split':
                                        x_train, x_test, y_train, y_test = train_test_split(x,
                                                                                            y,
                                                                                            test_size=sk_eval_value,
                                                                                            stratify=y,
                                                                                            random_state=CONFIG.getfloat('VARIABLES', 'RandomSeed'))
                                        preds = np.empty(len(y_test))
                                    else:
                                        self._update_log(
                                            f'No evaluation type chosen.')
                                except(KeyboardInterrupt, SystemExit):
                                    raise
                                except Exception:
                                    self.logger.warning(
                                        '{} threw an exception during fit. \
                                            Possible error with joblib multithreading.'.format(model), exc_info=True)
                                    tb = traceback.format_exc()
                                    print(tb)
                                    self._update_log('{} threw an exception during fit. \
                                            Possible error with joblib multithreading.'.format(model), True, False)
                                model_scores = self.get_model_scores(y, preds)

                                self._update_log(
                                    f'Task completed on <b>{model}</b>.')
                                table_str = '''<table>
                                                    <thead>
                                                        <tr>
                                                            <th>Accuracy</th><th>F1-Score</th><th>Cohen's Kappa</th>
                                                        </tr>
                                                    </thead>
                                                <tbody>
                                                    <tr>
                                            '''
                                for metric, score in model_scores.items():
                                    table_str += '<td style="border: 1px solid #333;">%.2f</td>' % score
                                table_str += '</tr></tbody></table><br>'
                                if sk_eval_type is not None:
                                    self._update_log(table_str, False, True)
                                self._update_log(
                                    f'Training {model} on full dataset')
                                with joblib.parallel_backend('dask'):
                                    pipeline.fit(x, y)

                                pred_col_name = col_label + TAG_DELIMITER + model + PRED_LABEL_SUFFIX
                                prob_col_name = col_label + TAG_DELIMITER + model + PROB_LABEL_SUFFIX
                                results[pred_col_name] = preds.astype(int)
                                # If predicting probabilities and the probability array has values,
                                # use those values for the results.
                                if self.use_proba and probs.size:
                                    results[prob_col_name] = np.amax(
                                        probs, axis=1)

                                save_path = os.path.join(col_path, model)
                                if not os.path.exists(save_path):
                                    os.makedirs(save_path)
                                self.save_model(
                                    model, pipeline, save_path, model_scores)
                            except (KeyboardInterrupt, SystemExit):
                                raise
                            except Exception as e:
                                self.logger.error(
                                    f'ModelTrainer.run {model}:', exc_info=True)
                                tb = traceback.format_exc()
                                print(tb)
                                self._update_log(tb)
                    # Tensorflow__ would reside here
                    try:
                        if self.train_stacking_algorithm and self._is_running:
                            self.train_stacker(results.drop(TRUTH_LABEL_SUFFIX, axis=1),
                                               results[TRUTH_LABEL_SUFFIX].values,
                                               col_path)
                        else:
                            self._update_log('Skipping Stacker training.')
                    except ValueError as ve:
                        self.signals.training_complete.emit(pd.DataFrame())
                        self._update_log(
                            f'Unable to train Stacking algorithm on {col_label}.')
                        tb = traceback.format_exc()
                        print(tb)
                    except Exception as e:
                        self.logger.error(
                            f'ModelTrainer.run {model}:', exc_info=True)
                        tb = traceback.format_exc()
                        print(tb)
                        self._update_log(tb)
            self._is_running = False
            self.signals.training_complete.emit(self.all_predictions_df)

        except Exception as e:
            self.signals.training_complete.emit(pd.DataFrame())
            self.logger.error('ModelTrainer.run (General):', exc_info=True)
            tb = traceback.format_exc()
            print(tb)
            self._update_log(tb)

    def get_model_scores(self, y, y_hat):
        '''
            Generate scores for a given model
                # Arguments
                    y: list, ground truth for a given classification task
                    y_hat: list, predictions generated by the model
                # Returns
                    scores: dict, generated scores.  Key is metric name and value is score
        '''
        scores = {}
        try:
            scores['accuracy'] = accuracy_score(y, y_hat)
            scores['f1_score'] = f1_score(y, y_hat, average='weighted')
            scores['cohen_kappa'] = cohen_kappa_score(y, y_hat)
        except ValueError as ve:
            self._update_log(
                "Unable to generate performance metrics.  Returning all values as zero.")
            scores['accuracy'] = 0
            scores['f1_score'] = 0
            scores['cohen_kappa'] = 0
        except Exception as e:
            # self.signals.training_complete.emit(0, False)
            self.logger.error('ModelTrainer.get_model_scores:', exc_info=True)
            tb = traceback.format_exc()
            print(tb)
            self._update_log(tb)

        return scores

    def get_params_from_file(self, model_name, base_path=None, tpot=False):
        '''
            Loads model parameters either from file (if version has been saved), or grabs the defaults
                # Arguments
                    model_name: string, model name used to specify path
                    base_path: string, optional pathing used for loading custom model parameters
                # Returns
                    model_params: dict, parameters from file or defaults
        '''
        try:
            if tpot or base_path is not None:
                model_path = os.path.join(
                    base_path, model_name, model_name + '.json')
                if not os.path.isfile(model_path):
                    model_path = os.path.join(CONFIG.get('PATHS', 'DefaultModelDirectory'),
                                              model_name,
                                              model_name + '.json')

            # elif base_path is not None:
            #     model_path = os.path.join(
            #         base_path, model_name, model_name + '.json')
            #     if not os.path.isfile(model_path):
            #         model_path = os.path.join(CONFIG.get('PATHS', 'DefaultModelDirectory'),
            #                                   model_name,
            #                                   model_name + '.json')
            else:
                model_path = os.path.join(CONFIG.get('PATHS', 'DefaultModelDirectory'),
                                          model_name,
                                          model_name + '.json')

            with open(model_path, 'r') as param_file:
                model_params = json.load(param_file, object_hook=cat_decoder)
            return model_params
        except Exception as e:
            self.logger.error(
                'ModelTrainer.get_params_from_file:', exc_info=True)
            tb = traceback.format_exc()
            print(tb)
            self._update_log(tb, True, False)

    def get_pipeline(self, param_dict, include_feature_selection=True):
        '''Builds pipeline steps required for sklearn models.  
            Includes Feature extraction, feature selection, and classifier.
                # Arguments
                    param_dict: dict, dictionary of current model parameter values.
                # Returns
                    pipeline_steps: list, list of steps with intialized classes.
        '''
        pipeline_queue = PriorityQueue()
        for args, values in param_dict.items():
            full_class = args.split('.')
            current_module = '.'.join(full_class[0:-1])
            current_type = full_class[1]

            if current_type == 'feature_extraction':
                priority = 0
            elif current_type == 'feature_selection':
                if include_feature_selection:
                    priority = 50
                else:
                    continue
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

    def get_tpot_pipeline(self, param_dict, tpot_params, include_feature_selection=False):
        pipeline_queue = PriorityQueue()
        for args, values in param_dict.items():
            full_class = args.split('.')
            current_module = '.'.join(full_class[0:-1])
            current_type = full_class[1]

            if current_type == 'feature_extraction':
                priority = 0
            elif current_type == 'feature_selection':
                if include_feature_selection:
                    priority = 50
                else:
                    continue
            else:
                continue
            inst_module = importlib.import_module(current_module)
            current_class = getattr(inst_module, full_class[-1])
            if values:
                pipeline_queue.put(
                    (priority, (full_class[-1], current_class(**values))))
            else:
                pipeline_queue.put(
                    (priority, (full_class[-1], current_class())))

        pipeline_queue.put(
            (100, ('TPOTClassifier', TPOTClassifier(**tpot_params['tpot.TPOTClassifier']))))

        pipeline = []
        while not pipeline_queue.empty():
            pipeline.append(pipeline_queue.get()[-1])
        return pipeline

    def grid_search(self,
                    model,
                    x,
                    y,
                    pipeline,
                    tuning_params,
                    n_jobs=-1,
                    n_iter=20,
                    scoring=None,
                    include_tfidf=False,
                    keras_params=None):
        '''Performs grid search on selected pipeline.

            # Arguments

                model: string, name of classifier in pipeline
                x: pandas.DataFrame, training data
                y: numpy.array, training labels
                pipeline: sklearn.model_selection.Pipeline, pipeline object containing feature extractors, feature selectors and estimator
                n_jobs: int, Number of jobs to run in parallel.
                n_iter: int, number of iterations to perform search
                scoring: list, scoring metrics to be used by the evaluator
                include_tfidf: bool:, flag to indicate tfidf is included in the pipeline
                keras_params: dict, parameters necessary for model training outside of the regular hyperparams.  e.g. input_shape, num_classes, num_features
        '''
        try:
            start_time = time.time()
            filepath = os.path.join(CONFIG.get(
                'PATHS', 'BaseModelDirectory'), model + '.json')
            with open(filepath, 'r') as f:
                model_data = json.load(f, object_hook=cat_decoder)

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
                with open(CONFIG.get('PATHS', 'BaseTfidfDirectory'), 'r') as f:
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
            
            self._update_log(f'Beginning RandomizedSearchCV on {model}...')
            rscv = RandomizedSearchCV(pipeline,
                                      grid_params,
                                      n_jobs=tuning_params['gridsearch']['n_jobs'] if tuning_params[
                                          'gridsearch']['n_jobs'] != 0 else None,
                                      cv=tuning_params['gridsearch']['cv'],
                                      n_iter=n_iter,
                                      pre_dispatch=CONFIG.get(
                                          'VARIABLES', 'PreDispatch'),
                                      verbose=CONFIG.getint(
                                          'VARIABLES', 'RandomizedSearchVerbosity'),
                                      scoring=tuning_params['gridsearch']['scoring'] if len(
                                          tuning_params['gridsearch']['scoring']) > 0 else 'accuracy',
                                      refit='accuracy')
            #   refit='accuracy' if len(tuning_params['gridsearch']['scoring']) > 0 else None)  # ! FIXME: Should we allow other, non accuracy metrics here?
            with joblib.parallel_backend('dask'):
                rscv.fit(x, y)
            self.grid_search_time = time.time() - start_time
            self._update_log(
                f'RandomizedSearchCV on {model} completed in {self.grid_search_time}')
            self._update_log(
                f'Best score for {model}: {rscv.best_score_}', False)
            return rscv

        except FileNotFoundError as fnfe:
            self.logger.debug(
                'ModelTrainer.grid_search {} not found'.format(filepath))
        except Exception as e:
            self.logger.error(
                'ModelTrainer.grid_search {}:'.format(model), exc_info=True)
            tb = traceback.format_exc()
            print(tb)
            self._update_log(tb)

    def save_model(self, model_name, pipeline, save_path, scores={}):
        save_file = os.path.join(
            save_path, model_name + '.pkl')
        self._update_log(
            f'Saving {model_name} to : {save_file}', False)
        joblib.dump(
            pipeline, save_file, compress=1)
        self.model_checksums[model_name] = hashlib.md5(
            open(save_file, 'rb').read()).hexdigest()
        self._update_log(
            f'{model_name} checksum: {self.model_checksums[model_name]}', False)
        if model_name == 'TPOTClassifier':
            self.save_tpot_params_to_file(pipeline, save_path, scores)
        else:
            self.save_params_to_file(
                model_name, pipeline.get_params(), save_path, scores)
        # if self.tune_models:
        #     if model_name == 'TPOTClassifier':
        #         self.save_tpot_params_to_file(pipeline, save_path, scores)
        #     else:
        #         self.save_params_to_file(
        #             model_name, pipeline.get_params(), save_path, scores)

    def save_params_to_file(self, model, best_params, model_param_path, score_dict={}):
        try:
            model_path = os.path.join(model_param_path, model + '.json')
            if not os.path.isfile(model_path):
                # Get default values
                model_path = os.path.join(CONFIG.get('PATHS', 'DefaultModelDirectory'),
                                          model,
                                          model + '.json')
            with open(model_path, 'r') as param_file:
                model_params = json.load(param_file)
            current_time = time.localtime()
            model_params['meta']['training_meta'].update(
                {
                    'last_train_date': time.strftime('%Y-%m-%d %H:%M:%S', current_time),
                    'train_eval_score': score_dict,
                    'checksum': self.model_checksums[model]
                }
            )
            if self.tune_models:
                model_params['meta']['tuning_meta'].update({
                    'last_tune_date': time.strftime('%Y-%m-%d %H:%M:%S', current_time),
                    'n_iter': self.tuning_params['gridsearch']['n_iter'],
                    'tuning_duration': self.grid_search_time,
                    'tune_eval_score': score_dict
                })

            # Update model params to those discovered during tuning
            for param_type, parameters in model_params['params'].items():
                param_key = param_type.split('.')[-1]
                for k, v in best_params.items():
                    best_param_key = k.split('__')[-1]
                    if k.startswith(param_key) and best_param_key in parameters.keys():
                        parameters[best_param_key] = v
            save_path = os.path.join(model_param_path, model + '.json')
            # print(f'Saving {model} params: {model_params} to {save_path}')
            with open(save_path, 'w') as outfile:
                json.dump(model_params, outfile, indent=2, cls=CATEncoder)

        except FileNotFoundError as fnfe:
            self.logger.debug(
                'ModelTrainer.save_params_to_file {} not found'.format(model_path))
        except Exception as e:
            self.logger.error(
                'ModelTrainer.save_params_to_file {}:'.format(model), exc_info=True)
            tb = traceback.format_exc()
            print(tb)

    def save_tpot_params_to_file(self, pipeline, model_param_path, score_dict):
        try:
            model = 'TPOTClassifier'
            model_path = os.path.join(model_param_path, model + '.json')
            if not os.path.isfile(model_path):
                # Get default values
                model_path = os.path.join(CONFIG.get('PATHS', 'DefaultModelDirectory'),
                                          model,
                                          model + '.json')
            with open(model_path, 'r') as param_file:
                model_params = json.load(param_file)

            best_params = pipeline.get_params()

            tpot_params = model_params['tpot_params']
            # * Remove any models under params that are not TfidfVectorizers
            for param_type in list(model_params['params'].keys()):
                param_key = param_type.split('.')[1]
                if param_key != 'feature_extraction':
                    del model_params['params'][param_type]

            # * Update tfidf params to the best
            for param_type, parameters in model_params['params'].items():
                param_key = param_type.split('.')[-1]
                for k, v in best_params.items():
                    best_param_key = k.split('__')[-1]
                    if k.startswith(param_key) and best_param_key in parameters.keys():
                        parameters[best_param_key] = v
            current_time = time.localtime()
            model_params['meta']['training_meta'].update({
                'last_train_date': time.strftime('%Y-%m-%d %H:%M:%S', current_time),
                'train_eval_score': score_dict,
                'checksum': self.model_checksums[model]
            })

            if self.tune_models:
                model_params['meta']['tuning_meta'].update({
                    'last_tune_date': time.strftime('%Y-%m-%d %H:%M:%S', current_time),
                    'n_iter': self.tuning_params['gridsearch']['n_iter'],
                    'tuning_duration': self.grid_search_time,
                    'tune_eval_score': score_dict
                })
            # * Now to get the new model parameters
            for name, obj in pipeline.named_steps.items():
                if name == 'TfidfVectorizer':
                    continue
                module_name = str(obj.__class__).split("'")[1]
                module_params = obj.get_params()
                model_params['params'].update({module_name: module_params})

            model_params['tpot_params'] = tpot_params

            with open(os.path.join(model_param_path, model + '.json'), 'w') as outfile:
                json.dump(model_params, outfile, indent=2, cls=CATEncoder)

        except FileNotFoundError as fnfe:
            self.logger.debug(
                'ModelTrainer.save_params_to_file {} not found'.format(model_path))
        except Exception as e:
            self.logger.error(
                'ModelTrainer.save_params_to_file {}:'.format(model), exc_info=True)
            tb = traceback.format_exc()
            print(tb)

    # @pyqtSlot()

    def stop_thread(self):
        self._update_log(
            'Attempting to stop ModelTrainer.<br>Current task must complete before stopping...')
        self._is_running = False

    def train_stacker(self, x, y, col_path):

        def get_ratio(row):
            """
            Returns the ratio of agreement between column values (here, predictors) in a given row.
            """
            try:
                pred_value = row.iloc[-1]
                total_same = 0.0
                col_count = float(len(row.iloc[:-1]))
                for data in row.iloc[:-1]:
                    if data == pred_value:
                        total_same += 1.0
                return total_same / col_count
            except ZeroDivisionError as zde:
                return 0
            except Exception as e:
                self.logger.error(
                    "ModelTrainer.get_ratio", exc_info=True)
                exceptionWarning(
                    'Exception occured in ModelTrainer.get_ratio.', repr(e))

        def get_bamboozled_score(row):
            """
            Returns the difference between the number of models and the number of models who predicted incorrectly.
            The higher this value, the more bamboozling the sample
            """
            try:
                pred_value = row.iloc[-1]
                total_wrong = 0
                col_count = len(row.iloc[:-1])
                for data in row.iloc[:-1]:
                    if data != pred_value:
                        total_wrong += 1
                return col_count - total_wrong
            except Exception as e:
                self.logger.error(
                    "ModelTrainer.get_bamboozled_score", exc_info=True)
                exceptionWarning(
                    'Exception occured in ModelTrainer.get_bamboozled_score.', repr(e))

        stacker_full_class = CONFIG.get(
            'VARIABLES', 'StackingAlgorithmCLassName').split('.')

        final_preds = np.empty(y.shape)
        stacker_module = '.'.join(stacker_full_class[0:-1])
        inst_module = importlib.import_module(stacker_module)
        stacker_class = getattr(inst_module, stacker_full_class[-1])
        stacker = stacker_class()
        if self.tuning_params['gridsearch']['tune_stacker']:
            self._update_log(
                f'Beginning tuning run on Stacker <b>{".".join(stacker_full_class)}</b>...')
            distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])
            rscv = RandomizedSearchCV(estimator=stacker,
                                      param_distributions=distributions,
                                      n_jobs=self.tuning_params['gridsearch']['n_jobs'] if self.tuning_params[
                                          'gridsearch']['n_jobs'] != 0 else None,
                                      cv=self.tuning_params['gridsearch']['cv'],
                                      n_iter=self.tuning_params['gridsearch']['n_iter'],
                                      pre_dispatch=CONFIG.get(
                                          'VARIABLES', 'PreDispatch'),
                                      verbose=CONFIG.getint(
                                          'VARIABLES', 'RandomizedSearchVerbosity'),
                                      scoring=self.tuning_params['gridsearch']['scoring'] if len(
                                          self.tuning_params['gridsearch']['scoring']) > 0 else 'accuracy',
                                      refit='accuracy')
            rscv.fit(x, y)
            best_params = rscv.best_params_
            stacker = stacker_class(**best_params)
            self._update_log('Stacker tuning completed!  Re-evaluating...')

        self._update_log(
            f'Training Stacking algorithm <b>{".".join(stacker_full_class)}</b>')
        skf = StratifiedKFold(n_splits=5,
                              random_state=RANDOM_SEED,
                              shuffle=True)

        for train, test in skf.split(x, y):
            with joblib.parallel_backend('dask'):
                stacker.fit(x.iloc[train], y[train])
            final_preds[test] = stacker.predict(x.iloc[test])
        # stack_preds = [1 if x > .5 else 0 for x in np.nditer(final_preds)]
        self._update_log('Stacking training complete')
        stack_scores = self.get_model_scores(y, final_preds)

        table_str = '''<table>
                            <thead>
                                <tr>
                                    <th>Accuracy</th><th>F1-Score</th><th>Cohen's Kappa</th>
                                </tr>
                            </thead>
                        <tbody>
                            <tr>
                    '''
        for metric, score in stack_scores.items():
            table_str += '<td style="border: 1px solid #333;">%.2f</td>' % score
        table_str += '</tr></tbody></table><br>'
        self._update_log(table_str, False, True)
        self._update_log('Retraining Stacker on full dataset')
        stacker.fit(x, y)
        save_path = os.path.join(col_path, 'Stacker')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = os.path.join(
            save_path, 'Stacker.pkl')
        self._update_log(f'Saving Stacking algorithm to : {save_file}', False)
        joblib.dump(stacker, save_file, compress=1)
        self.model_checksums['Stacker'] = hashlib.md5(
            open(save_file, 'rb').read()).hexdigest()
        self._update_log(f'Stacking hash: {self.model_checksums["Stacker"]}')

        # Save particulars to file
        col_name = col_path.split('\\')[-1]
        stacker_info = {
            'column': col_name,
            'version_directory': self.version_directory,
            'last_train_date': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            'train_eval_score': stack_scores,
            'model_checksums': self.model_checksums
        }
        stacker_json_save_file = os.path.join(save_path, 'Stacker.json')
        with open(stacker_json_save_file, 'w') as outfile:
            json.dump(stacker_info, outfile, indent=2)
        x[col_name + TRUTH_LABEL_SUFFIX] = y
        agreement_ratios = x.apply(get_ratio, axis=1)
        # bamboozled = x.apply(get_bamboozled_score, axis=1)

        x[col_name + TAG_DELIMITER + 'agreement_ratio'] = agreement_ratios
        # x[col_name + TAG_DELIMITER + 'bamboozled_score'] = bamboozled
        pc_len = len(x[x[col_name + TAG_DELIMITER + 'agreement_ratio'] <= DISAGREEMENT_THRESHOLD])
        # bamboozled_len = len(x[x[col_name + TAG_DELIMITER + 'bamboozled_score'] <= BAMBOOZLED_THRESHOLD])
        self._update_log(
            f"Found {pc_len} samples for {col_name} that fall at or below the {DISAGREEMENT_THRESHOLD} predictor agreement.")
        # self._update_log(
            # f"Found {bamboozled_len} samples for {col_name} that have a bamboozled score of {BAMBOOZLED_THRESHOLD} or below.")
        # print('HEAD OF X IN TRAIN_STACKER')
        # print(x.head())
        # print(x.columns)
        # ? What X is a dataframe  [col_name + CONFIG.get('VARIABLES', 'StackerLabelSuffix')] = final_preds
        self.all_predictions_df = pd.merge(self.all_predictions_df, x, how='outer', left_index=True, right_index=True)
        # print('HEAD OF all_redictions_df IN TRAIN_STACKER')
        # print(self.all_predictions_df.head())
        # print(self.all_predictions_df.columns)
        self._update_log('Run complete')
        self._update_log('<hr>', False, True)

    def _tune_model(self, x, y, model, col_path):
        model_params = self.get_params_from_file(
            model, col_path, True)
        if(model.lower() == 'tpotclassifier'):
            self._update_log(
                'Begin TPOT Optimization')
            tpot_pipeline = Pipeline(self.get_tpot_pipeline(
                model_params['params'], model_params['tpot_params']))
            with joblib.parallel_backend('dask'):
                tpot_pipeline.fit(x, y)
            new_steps = []
            new_steps.append(
                ('TfidfVectorizer', tpot_pipeline.named_steps['TfidfVectorizer']))
            fitted_pipeline = tpot_pipeline.named_steps[
                'TPOTClassifier'].fitted_pipeline_
            for n, p in fitted_pipeline.named_steps.items():
                new_steps.append((n, p))
            pipeline = Pipeline(new_steps)
        else:
            gs_pipeline = Pipeline(
                self.get_pipeline(model_params['params'], include_feature_selection=False))
            self._update_log(
                f'Begin tuning on {model}')
            with joblib.parallel_backend('dask'):
                pipeline = self.grid_search(model=model,
                                            x=x,
                                            y=y,
                                            pipeline=gs_pipeline,
                                            tuning_params=self.tuning_params,
                                            n_iter=self.tuning_params['gridsearch']['n_iter'],
                                            n_jobs=self.tuning_params['gridsearch']['n_jobs'],
                                            include_tfidf=True).best_estimator_

        if pipeline is None:
            self._update_log(
                f'Grid search failed for {model} on task {col_path}.  Skipping...')
            return False
        else:
            save_path = os.path.join(col_path, model)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            self.save_model(model, pipeline, save_path)
            return True

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
                'ModelTrainer._generate_best_param_dict {}:'.format(e), exc_info=True)
            tb = traceback.format_exc()
            print(tb)

    def _update_log(self, msg, include_time=True, as_html=True):
        # outbound = f'{time.strftime('%Y-%m-%d %H:%M:%S', current_time)} - {msg}<br>'
        self.signals.update_training_logger.emit(msg, include_time, as_html)
