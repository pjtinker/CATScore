from PyQt5.QtCore import QObject, Qt, QThread, QThreadPool, pyqtSignal, pyqtSlot, QSize, QModelIndex
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import (QAction, QButtonGroup, QCheckBox, QComboBox,
                             QDoubleSpinBox, QFileDialog, QFormLayout,
                             QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                             QMessageBox, QPushButton, QRadioButton,
                             QScrollArea, QSizePolicy, QSpinBox, QTabWidget, QHeaderView,
                             QVBoxLayout, QPlainTextEdit, QWidget, QTableView)
import json
import logging
import os
import traceback
import time
from collections import OrderedDict
from functools import partial
import hashlib

from chardet.universaldetector import UniversalDetector
import pandas as pd
import pkg_resources

from package.evaluate.EvaluateTableModel import EvaluateTableModel
from package.utils.catutils import exceptionWarning, clearLayout
from package.utils.DataframeTableModel import DataframeTableModel
from package.utils.GraphWidget import GraphWidget
from package.utils.config import CONFIG

from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score


class Communicate(QObject):
    version_change = pyqtSignal(str)    
    enable_eval_btn = pyqtSignal(bool)
    # stop_training = pyqtSignal()
    data_load = pyqtSignal(pd.DataFrame)
    update_statusbar = pyqtSignal(str)
    update_progressbar = pyqtSignal(int, bool)

class EvaluateWidget(QWidget):
    def __init__(self, parent=None):
        super(EvaluateWidget, self).__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.parent = parent
        self.comms = Communicate()

        self.prediction_data = pd.DataFrame()
        self.pc = pd.DataFrame()
        self.columns_with_truth = []
        
        self.open_file_button = QPushButton('Load CSV', self)
        self.open_file_button.clicked.connect(lambda: self.open_file())
        
        self.main_layout = QHBoxLayout()
        self.left_column = QVBoxLayout()
        self.right_column = QVBoxLayout()

        self.main_layout.addLayout(self.left_column)
        self.main_layout.addLayout(self.right_column)
        # * Available question column view
        self.available_column_view = QTableView()
        self.available_column_view.setMinimumHeight(321)
        self.available_column_view.setMaximumWidth(234)
        self.available_column_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.available_column_view.setSelectionMode(QTableView.SingleSelection)
        self.available_column_view.setSelectionBehavior(QTableView.SelectRows)
        self.available_column_model = EvaluateTableModel()
        self.available_column_view.setModel(self.available_column_model)
        self.available_column_view.setSpan(0, 0, 1, 2)
        selection = self.available_column_view.selectionModel()
        selection.selectionChanged.connect(
            lambda x: self.display_selected_rows(x))
        
        # Training stats and available models
        self.training_stats_groupbox = QGroupBox('Training Info')


        # self.training_stats_grid = QGridLayout()
        # self.training_stats_grid.setVerticalSpacing(0)
        # self.training_stats_groupbox.setLayout(self.training_stats_grid)
        # # self.training_stats_groupbox.setMinimumHeight(200)
        # model_label = QLabel("Model")
        # model_label.setFont(QFont("Times", weight=QFont.Bold))
        # self.training_stats_grid.addWidget(model_label, 0, 0, Qt.AlignTop)
        # train_date_label = QLabel("Last Trained")
        # train_date_label.setFont(QFont("Times", weight=QFont.Bold))
        # self.training_stats_grid.addWidget(train_date_label, 0, 1, Qt.AlignTop)
        # accuracy_label = QLabel("Accuracy")
        # accuracy_label.setFont(QFont("Times", weight=QFont.Bold))
        # self.training_stats_grid.addWidget(accuracy_label, 0, 2, Qt.AlignTop)
        # # self.training_stats_groupbox.setAlignment(Qt.AlignTop)
        # self.right_column.addWidget(self.training_stats_groupbox)

        self.training_stats_grid = QGridLayout()
        self.training_stats_grid.setVerticalSpacing(0)
        self.training_stats_groupbox.setLayout(self.training_stats_grid)
        self.training_stats_groupbox.setMinimumHeight(200)
        model_label = QLabel("Model")
        model_label.setFont(QFont("Times", weight=QFont.Bold))
        self.training_stats_grid.addWidget(model_label, 0, 0, Qt.AlignTop)
        train_date_label = QLabel("Last Trained")
        train_date_label.setFont(QFont("Times", weight=QFont.Bold))
        self.training_stats_grid.addWidget(train_date_label, 0, 1, Qt.AlignTop)
        accuracy_label = QLabel("Accuracy")
        accuracy_label.setFont(QFont("Times", weight=QFont.Bold))
        self.training_stats_grid.addWidget(accuracy_label, 0, 2, Qt.AlignTop)
        # self.training_stats_groupbox.setAlignment(Qt.AlignTop)
        self.right_column.addWidget(self.training_stats_groupbox)
        
        
        # Text DataframeTableModel view for text preview
        self.text_table_view = QTableView()
        self.text_table_view.setSelectionMode(QTableView.SingleSelection)
        self.text_table_view.setSelectionBehavior(QTableView.SelectRows)
        self.text_table_model = DataframeTableModel()
        self.text_table_view.setModel(self.text_table_model)
        self.text_table_view.setMaximumHeight(171)

        self.graph = GraphWidget(self, width=6, height=6, dpi=100)
        self.right_column.addWidget(self.graph)

        self.left_column.addWidget(self.open_file_button)
        self.left_column.addWidget(self.available_column_view)

        self.setLayout(self.main_layout)


    def display_selected_rows(self, selection=None):
        """
        Updates the stats and label distro plot when a question is selected.
            # Attributes
                selection: QItemSelectionModel, item currently selected by user.
        """
        
        if selection:
            idx = selection.indexes()[0]
        else:
            #* If no question selected, select the first in the list
            self.available_column_view.selectRow(0)
            self.available_column_view.setFocus()
            idx = QModelIndex(self.available_column_model.index(0, 0))

        col_name = self.available_column_model.data(idx)
        if self.available_column_model.getTruth(col_name):
            col_tag = col_name.split('__')[0]
            pred_col = col_tag + '__predicted'
            truth_col = col_tag + '__actual'
            try:
                preds = self.prediction_data[pred_col].values.astype(int)
                truth = self.prediction_data[truth_col].values.astype(int)
            except KeyError as ke:
                self.graph.clear_graph()
                return
            except Exception:
                raise
            f1 = f1_score(truth, preds, average='weighted')
            print(f1)
            acc = accuracy_score(truth, preds)
            print(acc)
            kappa = cohen_kappa_score(truth, preds)
            print(kappa)

            try:
                self.graph.plot_confusion_matrix(truth, preds)
            except Exception as e:
                self.logger.error(
                    "EvaluateWidget.display_selected_row", exc_info=True)
                tb = traceback.format_exc()
                print(tb)

        
    def open_file(self):
        """
        Open file chooser for user to select the CSV file containing their data
        Only CSV files are allowed at this time.
        """
        self.column_checkboxes = []
        self.selected_column_targets = []
        file_name, filter = QFileDialog.getOpenFileName(
            self, 'Open CSV', os.getenv('HOME'), 'CSV(*.csv)')
        if file_name:
            self.load_file(file_name)


    def load_file(self, f_path):
    
        """
        Load data from a CSV file to the workspace.
        Column 0 is used for the index column.
        chardet attempts to determine encoding if file is not utf-8.
            # Attributes
                f_path(String): The filename selected via open_file
        """
        # FIXME: Reset status bar when new data is loaded.
        try:
            self.graph.clear_graph()
            self.available_column_model.loadData([], include_labels=False)
            self.prediction_data = pd.read_csv(f_path, encoding='utf-8', index_col=0) #TODO: user defined index column
        except UnicodeDecodeError as ude:
            self.logger.warning(
                "UnicodeDecode error opening file", exc_info=True)
            self.comms.update_statusbar.emit(
                "Attempting to determine file encoding...")
            detector = UniversalDetector()
            try:
                for line in open(f_path, 'rb'):
                    detector.feed(line)
                    if detector.done:
                        break
                detector.close()
                print("chardet determined encoding type to be {}".format(
                    detector.result['encoding']))
                self.prediction_data = pd.read_csv(
                    f_path, encoding=detector.result['encoding'], index_col=0)
            except Exception as e:
                self.logger.error("Error detecing encoding", exc_info=True)
                exceptionWarning("Exception has occured.", exception=e)
        except IOError as ioe:
            self.logger.error("IOError detecting encoding", exc_info=True)
            exceptionWarning(
                "IO Exception occured while opening file.", exception=ioe)
        except Exception as e:
            self.logger.error("Error detecting encoding", exc_info=True)
            exceptionWarning("Error occured opening file.", exception=e)

        try:
            columns = self.prediction_data.columns
            self.available_columns = []
            self.columns_with_truth = []
            
            self.ground_truth_columns = self.prediction_data.columns[
                ~self.prediction_data.isna().any()].tolist()
            
            for column in columns:
                if column.lower().endswith("text"):
                    self.available_columns.append(column)
                    column_tag = column.split('__')[0]
                    if(column_tag + '__actual' in self.ground_truth_columns):
                        self.columns_with_truth.append(column)
                    
            if self.available_columns:
                self.available_column_model.loadData(
                    self.available_columns, include_labels=False)

            if self.columns_with_truth:
                self.available_column_model.setTruthData(
                    self.columns_with_truth)
                # self.full_text_count.setText(str(self.prediction_data.shape[0]))
                # self.display_selected_row(None)
                # self.select_all_btn.setEnabled(True)
                # self.deselect_all_btn.setEnabled(True)

            self.comms.update_statusbar.emit("CSV loaded.")
            # else:
            #     exceptionWarning("No allowable data discovered in file.")
        except pd.errors.EmptyDataError as ede:
            exceptionWarning('Empty Data Error.\n', exception=ede)
        except Exception as e:
            self.logger.error("Error loading dataframe", exc_info=True)
            exceptionWarning(
                "Exception occured.  PredictWidget.load_file.", exception=e)
            
    
    # def get_problem_children(self, column_tag, threshold=CONFIG.getfloat('VARIABLES', 'DisagreementThreshold')):
    #     """
    #     Displays the samples that fall below a particular threshold of agreement between classifiers.
        
    #         # Arguments
            
    #             column_tag, string: Column identifying tag
    #             threshold, float: threshold to determine if a sample is problematic.  Non-inclusive.
    #     """
    #     def get_ratio(row):
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
    #         # result_data['agreement_ratio'] = result_data.apply(get_ratio, axis=1)
    #         # print(result_data.head())
    #         col = column_tag + '__agreement_ratio'
    #         if(col in result.data.columns):
    #             self.pc = self.prediction_data[result_data[column_tag + '__agreement_ratio'] < threshold]
    #             self.text_table_model.loadData(self.pc)
    #         else:
    #             exceptionWarning(f"No agreement ratio found for {column_tag}")
            
    
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