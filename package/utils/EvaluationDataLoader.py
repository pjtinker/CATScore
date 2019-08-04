from PyQt5.QtCore import (QAbstractTableModel, QDateTime, QModelIndex,
                          Qt, QTimeZone, QByteArray, pyqtSignal, pyqtSlot, QThread)
from PyQt5.QtGui import QMovie, QIcon, QPixmap, QFont
from PyQt5.QtWidgets import (QAction, QGroupBox, QMessageBox, QCheckBox, QComboBox,
                             QApplication, QLabel, QFileDialog, QHBoxLayout, QVBoxLayout,
                             QGridLayout, QHeaderView, QProgressBar, QScrollArea,
                             QSizePolicy, QTableView, QWidget, QPushButton, QAbstractScrollArea)
import os
import logging
import traceback
from functools import partial
import sys
import json
import hashlib

import pandas as pd
from chardet.universaldetector import UniversalDetector

from package.utils.catutils import exceptionWarning, clearLayout, cat_decoder, CATEncoder
from package.utils.preprocess_text import processText, get_avg_words_per_sample
from package.utils.spellcheck import SpellCheck
from package.utils.DataframeTableModel import DataframeTableModel
from package.utils.AttributeTableModel import AttributeTableModel
from package.utils.GraphWidget import GraphWidget
"""EvaluationDataLoader imports CSV file and returns a dataframe with the appropriate columns.
For training data, DI will consider the nth column as a training sample
and nth+1 as ground truth.
CSV files must be formatted accordingly.
"""


class EvaluationDataLoader(QWidget):
    """
    TODO: Refactor this monstrosity into functions to setup UI
    """
    data_load = pyqtSignal(pd.DataFrame)
    update_statusbar = pyqtSignal(str)
    update_progressbar = pyqtSignal(int, bool)

    def __init__(self, parent=None):
        super(EvaluationDataLoader, self).__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.parent = parent
        self.column_checkboxes = []
        self.selected_column_targets = []
        self.text_preprocessing_checkboxes = []

        self.full_data = pd.DataFrame()
        self.selected_data = pd.DataFrame()
        self.allowable_columns = []
        self.trained_model_meta = {}
        
        self.version_selection_label = QLabel("Select version: ")
        self.version_selection = QComboBox(objectName='version_select')

        self.version_selection.addItem(
            'default', '.\\package\\data\\default_models\\default')
        available_versions = os.listdir(".\\package\\data\\versions")
        for version in available_versions:
            v_path = os.path.join('.\\package\\data\\versions', version)
            if os.path.isdir(v_path):
                self.version_selection.addItem(version, v_path)
        self.version_selection.currentIndexChanged.connect(lambda x, y=self.version_selection:
                                                           self._update_version(
                                                               y.currentData())
                                                           )
        self.open_file_button = QPushButton('Load CSV', self)
        self.open_file_button.clicked.connect(lambda: self.open_file())

        self.main_layout = QHBoxLayout()
        self.left_column = QVBoxLayout()
        self.right_column = QVBoxLayout()
        self.full_text_hbox = QHBoxLayout()

        # Column selection and basic stats
        self.text_stats_groupbox = QGroupBox("Selected Question")
        self.text_stats_grid = QGridLayout()

        self.full_text_count_label = QLabel("Total samples")
        self.full_text_count = QLabel()

        self.current_question_count_label = QLabel("No. samples")
        self.current_question_count = QLabel()
        self.current_question_avg_word_label = QLabel("Avg. words per sample")
        self.current_question_avg_word = QLabel()

        self.text_stats_grid.addWidget(self.current_question_count_label, 1, 0)
        self.text_stats_grid.addWidget(self.current_question_count, 1, 1)
        self.text_stats_grid.addWidget(
            self.current_question_avg_word_label, 2, 0)
        self.text_stats_grid.addWidget(self.current_question_avg_word, 2, 1)
        self.full_text_count.setText("None")
        self.text_stats_groupbox.setLayout(self.text_stats_grid)

        # ~ Available question column view
        self.available_column_view = QTableView()
        self.available_column_view.setMinimumHeight(330)
        self.available_column_view.setMaximumWidth(220)
        self.available_column_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.available_column_view.setSelectionMode(QTableView.SingleSelection)
        self.available_column_view.setSelectionBehavior(QTableView.SelectRows)
        self.available_column_model = AttributeTableModel()
        self.available_column_view.setModel(self.available_column_model)
        self.available_column_view.setSpan(0, 0, 1, 2)
        selection = self.available_column_view.selectionModel()
        selection.selectionChanged.connect(
            lambda x: self.display_selected_row(x))

        self.left_column.addWidget(self.version_selection)
        self.left_column.addStretch()
        self.left_column.addWidget(self.open_file_button)
        self.left_column.addWidget(self.available_column_view)

        self.load_data_btn = QPushButton('Load Data', self)
        self.load_data_btn.clicked.connect(lambda: self.load_selected_data())
        self.select_all_btn = QPushButton('Select All', self)
        self.select_all_btn.clicked.connect(
            lambda: self.available_column_model.setCheckboxes(True))
        self.select_all_btn.setEnabled(False)
        self.deselect_all_btn = QPushButton('Remove All', self)
        self.deselect_all_btn.clicked.connect(
            lambda: self.available_column_model.setCheckboxes(False))
        self.deselect_all_btn.setEnabled(False)

        self.selection_button_layout = QHBoxLayout()
        self.selection_button_layout.addWidget(self.select_all_btn)
        self.selection_button_layout.addWidget(self.deselect_all_btn)

        self.left_column.addLayout(self.selection_button_layout)
        self.left_column.addWidget(self.load_data_btn)
        # self.left_column.addStretch()

        # Text preprocessing options
        self.text_proc_groupbox = QGroupBox("Text Preprocessing Options")
        self.text_proc_groupbox.setEnabled(False)
        self.text_proc_grid = QGridLayout()
        self.text_proc_groupbox.setLayout(self.text_proc_grid)
        # self.data_load.connect(self.set_preprocessing_option_state)
        self.preprocess_text_btn = QPushButton('Preprocess Text', self)
        self.preprocess_text_btn.clicked.connect(
            lambda: self.applyPreprocessing())
        self.preprocess_text_btn.setEnabled(False)

        self.left_column.addWidget(self.text_proc_groupbox)
        self.left_column.addWidget(self.preprocess_text_btn)
        self.left_column.addStretch()

        # Data subset save button
        self.export_dataset_btn = QPushButton('&Export Data', self)
        icon = QIcon()
        icon.addPixmap(QPixmap('icons/Programming-Save-icon.png'))
        self.export_dataset_btn.setIcon(icon)
        self.export_dataset_btn.setEnabled(False)
        self.export_dataset_btn.clicked.connect(lambda: self.save_data())
        # self.export_dataset_btn.resize(32, 32)
        self.left_column.addWidget(self.export_dataset_btn)

        self.full_text_hbox.addWidget(self.text_stats_groupbox)
        self.full_text_hbox.addStretch()
        self.full_text_hbox.addWidget(self.full_text_count_label)
        self.full_text_hbox.addWidget(self.full_text_count)

        self.right_column.addLayout(self.full_text_hbox)

        # Training stats and available models
        self.training_stats_groupbox = QGroupBox('Training Info')
        self.training_stats_grid = QGridLayout()
        self.training_stats_groupbox.setLayout(self.training_stats_grid)
        model_label = QLabel("Model")
        model_label.setFont(QFont("Times", weight=QFont.Bold))
        self.training_stats_grid.addWidget(model_label, 0, 0)
        train_date_label = QLabel("Last Trained")
        train_date_label.setFont(QFont("Times", weight=QFont.Bold))
        self.training_stats_grid.addWidget(train_date_label, 0, 1)
        accuracy_label = QLabel("Accuracy")
        accuracy_label.setFont(QFont("Times", weight=QFont.Bold))
        self.training_stats_grid.addWidget(accuracy_label, 0, 2)
        self.right_column.addWidget(self.training_stats_groupbox)
        
        # Text DataframeTableModel view for text preview
        self.text_table_view = QTableView()
        self.text_table_view.setSelectionMode(QTableView.SingleSelection)
        self.text_table_view.setSelectionBehavior(QTableView.SelectRows)
        self.text_table_model = DataframeTableModel()
        self.text_table_view.setModel(self.text_table_model)
        self.text_table_view.setMinimumHeight(170)
        self.right_column.addWidget(self.text_table_view)
        # self.right_column.addStretch()
        self.main_layout.addLayout(self.left_column)
        # self.main_layout.addStretch()
        self.main_layout.addLayout(self.right_column)

        self.setup_text_preproc_ui()
        # Fire update version to set data in TableView
        self._update_version(self.version_selection.currentData())

        self.setLayout(self.main_layout)

    def _update_version(self, current_dir):
        row = 1
        column = 0
        try:
            for root, dirs, files in os.walk(current_dir):
                if 'Stacker.json' in files:
                    with open(os.path.join(root, 'Stacker.json')) as infile:
                        stacker_data = json.load(infile)
                        column_name = stacker_data['column'] + '_Text'
                        self.allowable_columns.append(column_name)
                        self.trained_model_meta[column_name] = {}
                        self.trained_model_meta[column_name]['Stacker'] = stacker_data
                        self._load_trained_models(os.path.split(root)[0], stacker_data['model_checksums'], column_name)
                   
            self.available_column_model.setAllowableData(self.allowable_columns)
        except Exception as e:
            self.logger.error(
                "EvaluationDataLoader._update_version", exc_info=True)
            print("Exception {}".format(e))
            tb = traceback.format_exc()
            print(tb)
    
    def _load_trained_models(self, model_dir, model_checksums, col_name):
        try:
            print(f"Path in _load_trained_models: {model_dir}")
            for model, checksum in model_checksums.items():
                if model == 'Stacker':
                    continue
                model_path = os.path.join(model_dir, model, model + '.pkl')
                print(f"model_path: {model_path}")
                current_chksum = hashlib.md5(open(model_path, 'rb').read()).hexdigest()
                if(current_chksum != checksum):
                    self.logger.warning(f"EvaluationDataLoader._load_training_models: \
                        Checksums for model {model_path} do not match.  \
                        Model checksum: {current_chksum}, Saved checksum: {checksum}")
                    exceptionWarning(f"Checksums for {model_path} are invalid.  Skipping... ")
                    continue
                model_param_path = os.path.join(model_dir, model, model + '.json')
                print(f"model_param_path: {model_param_path}")
                with open(model_param_path, 'r') as infile:
                    model_data = json.load(infile, object_hook=cat_decoder)
                self.trained_model_meta[col_name][model] = model_data['meta']['training_meta']
                
        except Exception as e:
            self.logger.error(
                "EvaluationDataLoader._load_trained_models", exc_info=True)
            print("Exception {}".format(e))
            tb = traceback.format_exc()
            print(tb)

    def load_selected_data(self):
        """Return columns selected from dataframe by user.
            # Returns
                list: column names selected by user
        """
        self.selected_columns = self.available_column_model.getChecklist()
        if len(self.selected_columns) == 0:
            self.text_proc_groupbox.setEnabled(False)
            self.selected_data = pd.DataFrame()
            self.text_table_model.loadData(None)
            self.preprocess_text_btn.setEnabled(False)
            self.export_dataset_btn.setEnabled(False)
            self.data_load.emit(pd.DataFrame())
            # exceptionWarning('No questions selected')
        else:
            self.selected_data = self.full_data[self.selected_columns].copy()
            self.text_table_model.loadData(self.selected_data.head())
            self.set_preprocessing_option_state(1, True)
            self.data_load.emit(self.selected_data)

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
            self.full_data = pd.read_csv(f_path, encoding='utf-8', index_col=0)
        except UnicodeDecodeError as ude:
            self.logger.warning(
                "UnicodeDecode error opening file", exc_info=True)
            self.update_statusbar.emit(
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
                self.full_data = pd.read_csv(
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
            columns = self.full_data.columns
            self.available_columns = []
            for column in columns:
                if column.endswith("Text"):
                    self.available_columns.append(column)
            self.available_column_model.loadData(
                self.available_columns, include_labels=False)

            self.available_column_model.setAllowableData(
                self.allowable_columns)
            self.full_text_count.setText(str(self.full_data.shape[0]))
            self.display_selected_row(None)
            self.select_all_btn.setEnabled(True)
            self.deselect_all_btn.setEnabled(True)

            self.update_statusbar.emit("CSV loaded.")
        except pd.errors.EmptyDataError as ede:
            exceptionWarning(
                exceptionTitle='Empty Data Error.\n', exception=ede)
        except Exception as e:
            self.logger.error("Error loading dataframe", exc_info=True)
            exceptionWarning(
                "Exception occured.  EvaluationDataLoader.load_file.", exception=e)

    def display_selected_row(self, selection=None):
        """
        Updates the stats and label distro plot when a question is selected.
            # Attributes
                selection: QItemSelectionModel, item currently selected by user.
        """
        try:
            if selection:
                idx = selection.indexes()[0]
            else:
                # If no question selected, select the first in the list
                self.available_column_view.selectRow(0)
                self.available_column_view.setFocus()
                idx = QModelIndex(self.available_column_model.index(0, 0))
            row = idx.row()
            col_name = self.full_data.columns[row]
            print(f"{row}, {col_name}")
            self.text_stats_groupbox.setTitle(col_name)
            question_data = self.full_data[self.full_data.columns[row]].fillna(
                value="unanswered")

            avg_num_words = get_avg_words_per_sample(str(question_data.values))
            self.current_question_count.setText(str(question_data.shape[0]))
            self.current_question_avg_word.setText("%.2f" % avg_num_words)
            
            grid_row = 1
            grid_column = 0
            print(f"col_name: {col_name}")
            print(f"from display_selected_rows: \ntrained_model_meta: {json.dumps(self.trained_model_meta, indent=2)}")
            clearLayout(self.training_stats_grid)
            model_label = QLabel("Model")
            model_label.setFont(QFont("Times", weight=QFont.Bold))
            self.training_stats_grid.addWidget(model_label, 0, 0)
            train_date_label = QLabel("Last Trained")
            train_date_label.setFont(QFont("Times", weight=QFont.Bold))
            self.training_stats_grid.addWidget(train_date_label, 0, 1)
            accuracy_label = QLabel("Accuracy")
            accuracy_label.setFont(QFont("Times", weight=QFont.Bold))
            self.training_stats_grid.addWidget(accuracy_label, 0, 2)
            for model, meta in self.trained_model_meta[col_name].items():

                print(f"params in meta for {model}: {json.dumps(meta, indent=2)}")
                self.training_stats_grid.addWidget(QLabel(model), grid_row, grid_column)
                grid_column += 1
                self.training_stats_grid.addWidget(QLabel(meta['last_train_date']), grid_row, grid_column)
                grid_column += 1
                self.training_stats_grid.addWidget(QLabel("%.2f" % meta['train_eval_score']), grid_row, grid_column)
                grid_row += 1
                grid_column = 0
            self.repaint()
        except Exception as e:
            self.logger.error("EvaluationDataLoader.display_selected_row", exc_info=True)
            exceptionWarning(
                "Exception occured.  EvaluationDataLoader.load_file.", exception=e)
            tb = traceback.format_exc()
            print(tb)


    def save_data(self):
        if self.selected_data.empty:
            exceptionWarning('No data selected')
            return
        file_name, filter = QFileDialog.getSaveFileName(
            self, 'Save to CSV', os.getenv('HOME'), 'CSV(*.csv)')
        if file_name:
            self.selected_data.to_csv(
                file_name, index_label='testnum', quoting=1, encoding='utf-8')
            self.update_statusbar.emit("Data saved successfully.")

    def setup_text_preproc_ui(self):
        """
        Generate necessary UI and backend data structures for text preprocessing option
        selection.
        """
        self.preprocessing_options = {
            "lower_case": True,
            "remove_punctuation": True,
            "expand_contractions": True,
            "remove_stopwords": False,
            "lemmatize": False,
            "spell_correction": False
        }
        proc_labels = [
            'Convert to lowercase', 'Remove punctuation',
            'Expand contractions', 'Remove stopwords',
            'Lemmatize'
        ]
        row = 0
        for k, v in self.preprocessing_options.items():
            chkbox = QCheckBox(' '.join(k.split('_')))
            chkbox.setChecked(v)
            chkbox.stateChanged.connect(lambda state, o=k, :
                                        self._update_preprocessing_options(
                                            o, state)
                                        )
            self.text_proc_grid.addWidget(chkbox, row, 0)
            self.text_preprocessing_checkboxes.append(chkbox)
            row = row + 1

    def _update_preprocessing_options(self, option, state):
        """
        Updates the selected text preprocessing options.
            # Attributes
                option: String, The text preprocessing option to update
                state: QtCore.Qt.CheckState, The state of the checkbox currently.  This value
                is either 0 or 2, (true, false) so we convert the 2 into a 1 if True.
        """
        truth = False
        if state != 0:
            truth = True
        self.preprocessing_options[option] = truth

    # @pyqtSlot(int, bool)
    def set_preprocessing_option_state(self, tab, state):
        """
        pyqtSlot for determining if text preprocessing options are selectable.  Based on
        if data has been successfully loaded and selected.  Reusing the pyqtSignal that
        enables the Model Selection tab, thus the tab attribute is irrelevant.
            # Attributes:
                tab: int, Tab to enable.  Irrelevant in this case.
                state: bool, The state of the preprocess options.
        """
        self.text_proc_groupbox.setEnabled(state)
        self.preprocess_text_btn.setEnabled(state)
        # if not state:
        #     self.text_proc_groupbox.setEnabled(False)
        #     self.preprocess_text_btn.setEnabled(False)
        #     #self.export_dataset_btn.setEnabled(False)
        # else:
        #     self.text_proc_groupbox.setEnabled(True)
        #     self.preprocess_text_btn.setEnabled(True)
        #     #self.export_dataset_btn.setEnabled(True)

    @pyqtSlot(pd.DataFrame)
    def update_data(self, data):
        self.load_data_btn.setEnabled(True)
        # self.text_proc_groupbox.setEnabled(True)
        # self.preprocess_text_btn.setEnabled(True)
        # self.export_dataset_btn.setEnabled(True)
        self.set_preprocessing_option_state(1, True)
        self.export_dataset_btn.setEnabled(True)

        self.update_statusbar.emit("Text preprocessing complete.")
        self.update_progressbar.emit(0, False)
        self.selected_data = data
        self.text_table_model.loadData(self.selected_data.head())
        self.data_load.emit(self.selected_data)

    def applyPreprocessing(self):
        """
        Spins up a thread to apply user selected preprocessing to input data.
        Data is stored as self.selected_data.

        """
        try:
            self.load_data_btn.setEnabled(False)
            self.text_proc_groupbox.setEnabled(False)
            self.preprocess_text_btn.setEnabled(False)
            self.export_dataset_btn.setEnabled(False)
            self.update_progressbar.emit(0, True)
            self.preproc_thread = PreprocessingThread(self.full_data[self.selected_columns],
                                                      self.preprocessing_options)
            self.preproc_thread.preprocessing_complete.connect(
                self.update_data)
            self.update_statusbar.emit(
                'Preprocessing text.  This may take several minutes.')
            self.preproc_thread.start()
        except Exception as e:
            self.logger.exception(
                "Exception occured in EvaluationDataLoader.applyPreprocessing", exc_info=True)
            tb = traceback.format_exc()
            print(tb)


class PreprocessingThread(QThread):
    """
    QThread to handle all text data preprocessing.
    This can be an expensive operation, especially if spell_correction is requested.
    """
    preprocessing_complete = pyqtSignal(pd.DataFrame)

    def __init__(self, data, options):
        """
        Make a new thread instance to apply text preprocessing to 
        the selected data.
            # Attributes:
                data: pandas.DataFrame, user selected columns for preprocessing
                options: dict, Text preprocessing options
        """
        super(PreprocessingThread, self).__init__()
        self.data = data
        self.options = options

    def run(self):
        # sys.stdout = open('nul', 'w')
        # print()
        print(self.options)
        apply_cols = [
            col for col in self.data.columns if col.endswith('_Text')
        ]
        self.data[apply_cols] = self.data[apply_cols].applymap(
            lambda x: processText(str(x), **self.options)
        )
        if self.options['spell_correction']:
            sentences = self.data[apply_cols].applymap(
                lambda x: str(x).split()
            ).values
            sc = SpellCheck(sentences, 5000)

            self.data[apply_cols] = self.data[apply_cols].applymap(
                lambda x: sc.correct_spelling(x)
            )
        # sys.stdout = sys.__stdout__
        self.preprocessing_complete.emit(self.data)

    def stop_thread(self):
        # TODO: Add funtionality to stop the thread
        self.quit()
        self.wait()