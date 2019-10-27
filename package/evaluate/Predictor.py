"""
QThread for model evaluation on new data.
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

import pandas as pd
import numpy as np

from sklearn.utils import parallel_backend, register_parallel_backend

import joblib
from joblib._parallel_backends import ThreadingBackend, SequentialBackend, LokyBackend
import scipy

from package.utils.catutils import CATEncoder, cat_decoder, exceptionWarning
# import package.utils.keras_models as keras_models
import package.utils.embedding_utils as embed_utils
from package.utils.config import CONFIG

TAG_DELIMITER = CONFIG.get('VARIABLES', 'TagDelimiter')
PRED_LABEL_SUFFIX = CONFIG.get('VARIABLES', 'PredictedLabelSuffix')
TRUTH_LABEL_SUFFIX = CONFIG.get('VARIABLES', 'TruthLabelSuffix')

STACKER_LABEL_SUFFIX = CONFIG.get('VARIABLES', 'StackerLabelSuffix')
DISAGREEMENT_THRESHOLD = CONFIG.getfloat('VARIABLES', 'DisagreementThreshold')
BAMBOOZLED_THRESHOLD = CONFIG.getint('VARIABLES', 'BamboozledThreshold')


class PredictorSignals(QObject):
    prediction_complete = pyqtSignal(pd.DataFrame)
    # tuning_complete = pyqtSignal(bool, dict)
    update_progressbar = pyqtSignal(int, bool)
    update_eval_logger = pyqtSignal(str)


class Predictor(QRunnable):
    """
    QThread tasked with evaluating data on previously trained models and saving results.
    """

    def __init__(self, model_metadata, evaluation_data):
        super(Predictor, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.signals = PredictorSignals()

        self.model_metadata = model_metadata
        self.evaluation_data = evaluation_data
        self.predictions = pd.DataFrame(index=self.evaluation_data.index)

    @pyqtSlot()
    def run(self):

        def get_agreement_ratio(row):
            """
            Returns the ratio of agreement between column values (here, predictors) in a given row.
            """
            pred_value = row.iloc[-1]
            total_same = 0.0
            col_count = float(len(row.iloc[:-1]))
            for data in row.iloc[:-1]:
                if data == pred_value:
                    total_same += 1.0
            return total_same / col_count

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

        self._update_log("Beginning Evaluation run...")
        meta_copy = self.model_metadata.copy()
        data_copy = pd.DataFrame(index=self.evaluation_data.index)

        for col_idx, column in enumerate(self.evaluation_data, 0):
            stacker_meta = meta_copy[column]['Stacker']
            column_prefix = column.split(TAG_DELIMITER)[0]
            last_trained = []
            self._update_log(f"Current evaluation task: {column_prefix}")
            for model_name, model_meta in meta_copy[column].items():
                if(model_name == 'Stacker'):
                    continue

                self._update_log(f"Evaluating via model: {model_name}")
                model = joblib.load(model_meta['model_path'])
                with joblib.parallel_backend('dask'):
                    self.predictions[column_prefix + TAG_DELIMITER +
                                     model_name + PRED_LABEL_SUFFIX] = model.predict(self.evaluation_data[column])
                last_trained.append(column_prefix + TAG_DELIMITER + model_name + PRED_LABEL_SUFFIX)

            self._update_log(f"Beginning Stacker evaluation")
            stacker = joblib.load(stacker_meta['model_path'])
            with joblib.parallel_backend('dask'):
                self.predictions[column_prefix +
                                 STACKER_LABEL_SUFFIX] = stacker.predict(self.predictions[last_trained])
            data_copy[column] = self.evaluation_data[column].copy()
            data_copy[column_prefix +
                      PRED_LABEL_SUFFIX] = self.predictions[column_prefix + STACKER_LABEL_SUFFIX]
            data_copy[column_prefix +
                      TRUTH_LABEL_SUFFIX] = np.NaN
            self._update_log(f"Searching for problematic samples for {column}")

            last_trained.append(column_prefix + STACKER_LABEL_SUFFIX)
            agreement_ratios =  self.predictions[last_trained].apply(get_agreement_ratio, axis=1)
            bamboozled_score = self.predictions[last_trained].apply(get_bamboozled_score, axis=1)
            self.predictions[column_prefix + '__agreement_ratio'] = agreement_ratios
            data_copy[column_prefix + '__agreement_ratio'] = bamboozled_score
            bamboozled_len = len(self.predictions[self.predictions[column_prefix + '__agreement_ratio'] <= BAMBOOZLED_THRESHOLD])
            pc_len = len(
                self.predictions[self.predictions[column_prefix + '__agreement_ratio'] <= DISAGREEMENT_THRESHOLD])
            self._update_log(
                f"Found {pc_len} samples for {column} that fall below {DISAGREEMENT_THRESHOLD} predictor agreement.")
            self._update_log(
                f"Found {bamboozled_len} samples for {column} that have a bamboozled score of {BAMBOOZLED_THRESHOLD} or below.")
            self._update_log(f"Evaluation for {column} complete.\n")

        self._update_log("Evaluation run complete.")
        result_dir = stacker_meta['version_directory']
        current_time = time.localtime()
        path_prefix = time.strftime('%Y-%m-%d_%H-%M', current_time)
        result_path = os.path.join(result_dir, 'results.csv')
        self._update_log(
            f'<b>Saving results to: <font color="#ffb900">{result_path}</font></b>')
        self.predictions.to_csv(
            result_path, index_label="testnum", encoding='utf-8')
        stacker_cols = [
            x for x in self.predictions.columns if x.endswith('Stacker')]
        # Delete Dataframe for space
        del self.predictions
        self.signals.prediction_complete.emit(data_copy)

    def _update_log(self, msg):
        # outbound = f"{time.ctime(time.time())} - {msg}<br>"
        self.signals.update_eval_logger.emit(msg)

    # def get_problem_children(self, threshold=0.5):
    #     """
    #     Find samples that fall below a defined threshold ratio in terms of prediction agreement between models.  

    #         # Arguments

    #             threshold, float: Value for which any sample's agreement falls below is marked as problematic.
    #     """
    #     def get_agreement_ratio(row):
    #         """
    #         Returns the ratio of agreement between column values (here, predictors) in a given row.
    #         """
    #         pred_value = row.iloc[-1]
    #         total_same = 0.0
    #         col_count = float(len(row.iloc[:-1]))
    #         for data in row.iloc[:-1]:
    #             if data == pred_value:
    #                 total_same += 1.0
    #         return total_same / col_count

    #     file_name, filter = QFileDialog.getOpenFileName(
    #         self, 'Open CSV', os.getenv('HOME'), 'CSV(*.csv)')
    #     if file_name:
    #         result_data = pd.read_csv(file_name, index_col=0)
    #         result_data['agreement_ratio'] = result_data.apply(
    #             get_agreement_ratio, axis=1)
    #         print(result_data.head())
    #         self.pc = self.prediction_data[result_data['agreement_ratio'] <= threshold]
    #         self.text_table_model.loadData(self.pc)

    # def save_pc(self):
    #     if self.pc.empty:
    #         exceptionWarning("No problem children loaded.")
    #         return

    #     file_name, filter = QFileDialog.getSaveFileName(
    #         self, 'Save to CSV', os.getenv('HOME'), 'CSV(*.csv)')
    #     if file_name:
    #         self.pc.to_csv(
    #             file_name, index_label='testnum', quoting=1, encoding='utf-8')
    #         self.comms.update_statusbar.emit("PC saved successfully.")
