from PyQt5.QtCore import (QAbstractTableModel, QDateTime, QModelIndex,
                          Qt, QTimeZone, QByteArray, pyqtSignal, pyqtSlot, QThread)
from PyQt5.QtGui import QMovie, QIcon, QPixmap
from PyQt5.QtWidgets import (QAction, QGroupBox, QMessageBox, QCheckBox, QApplication, QLabel, QFileDialog, QHBoxLayout, QVBoxLayout,
                             QGridLayout, QHeaderView, QProgressBar, QScrollArea, QSizePolicy, QTableView, QWidget, QPushButton)
import os
import logging
import traceback
from functools import partial
import sys

import pandas as pd
from chardet.universaldetector import UniversalDetector

from package.utils.catutils import exceptionWarning, clearLayout
from package.utils.preprocess_text import processText, get_avg_words_per_sample
from package.utils.spellcheck import SpellCheck
from package.utils.DataframeTableModel import DataframeTableModel
from package.utils.AttributeTableModel import AttributeTableModel
from package.utils.GraphWidget import GraphWidget
from package.utils.config import CONFIG

"""DataLoader imports CSV file and returns a dataframe with the appropriate columns.
For training data, DI will consider the nth column as a training sample
and nth+1 as ground truth.
CSV files must be formatted accordingly.
"""


class DataLoader(QWidget):
    """
    TODO: Refactor this monstrosity into functions to setup UI
    """
    data_load = pyqtSignal(pd.DataFrame)
    update_statusbar = pyqtSignal(str)
    update_progressbar = pyqtSignal(int, bool)

    def __init__(self, parent=None):
        super(DataLoader, self).__init__(parent)
        self.logger = logging.getLogger(__name__)
        # self.logger.setLevel(logging.DEBUG)
        self.parent = parent
        self.column_checkboxes = []
        self.selected_column_targets = []
        self.text_preprocessing_checkboxes = []

        self.full_data = pd.DataFrame()
        self.selected_data = pd.DataFrame()
        self.open_file_button = QPushButton('Load CSV', self)
        self.open_file_button.clicked.connect(lambda: self.open_file())

        self.main_layout = QHBoxLayout()
        self.left_column = QVBoxLayout()
        self.right_column = QVBoxLayout()
        self.full_text_hbox = QHBoxLayout()

        # Column selection and basic stats
        self.text_stats_groupbox = QGroupBox("Selected Question")
        self.text_stats_groupbox.setMinimumWidth(400)
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
        self.available_column_view.setMinimumHeight(322)
        self.available_column_view.setMaximumWidth(214)
        self.available_column_view.setSelectionMode(QTableView.SingleSelection)
        self.available_column_view.setSelectionBehavior(QTableView.SelectRows)
        self.available_column_model = AttributeTableModel()
        self.available_column_view.setModel(self.available_column_model)
        selection = self.available_column_view.selectionModel()
        selection.selectionChanged.connect(
            lambda x: self.display_selected_rows(x))

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
        self.export_dataset_btn.resize(32, 32)
        self.left_column.addWidget(self.export_dataset_btn)

        self.full_text_hbox.addWidget(self.text_stats_groupbox)
        self.full_text_hbox.addStretch()
        self.full_text_hbox.addWidget(self.full_text_count_label)
        self.full_text_hbox.addWidget(self.full_text_count)

        self.right_column.addLayout(self.full_text_hbox)
        # self.right_column.addWidget(self.text_stats_groupbox)
        self.graph = GraphWidget(self, width=6, height=6, dpi=100)
        self.right_column.addWidget(self.graph)

        # Text DataframeTableModel view for text preview
        self.text_table_view = QTableView()
        self.text_table_view.setSelectionMode(QTableView.SingleSelection)
        self.text_table_view.setSelectionBehavior(QTableView.SelectRows)
        self.text_table_model = DataframeTableModel()
        self.text_table_view.setModel(self.text_table_model)
        self.right_column.addWidget(self.text_table_view)

        self.main_layout.addLayout(self.left_column)
        # self.main_layout.addStretch()
        self.main_layout.addLayout(self.right_column)

        self.setup_text_preproc_ui()

        self.setLayout(self.main_layout)

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
            self.available_column_model.loadData([])
            self.select_all_btn.setEnabled(False)
            self.deselect_all_btn.setEnabled(False)
            self.full_data = pd.read_csv(f_path, encoding='utf-8', index_col=0)
        except UnicodeDecodeError as ude:
            self.logger.warning(
                "UnicodeDecode error opening file", exc_info=True)
            print("UnicodeDecodeError caught.  File is not UTF-8 encoded. \
                   Attempting to determine file encoding...")
            self.update_statusbar.emit(
                "File is not UTF-8 encoded. Attempting to determine file encoding...")
            detector = UniversalDetector()
            try:
                for line in open(f_path, 'rb'):
                    detector.feed(line)
                    if detector.done:
                        break
                detector.close()
                self.update_statusbar.emit("Chardet determined encoding type to be {}".format(
                    detector.result['encoding']))
                self.logger.info("Chardet determined encoding type to be {}".format(
                    detector.result['encoding']))
                self.full_data = pd.read_csv(
                    f_path, encoding=detector.result['encoding'], index_col=0)
            except Exception as e:
                self.logger.error("Error detecting encoding", exc_info=True)
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
                if column.endswith("__text"):
                    label_col = column.split('__')[0] + "__actual" 
                    if label_col in columns:
                        self.available_columns.append(column)
                        self.available_columns.append(label_col)
            # If no data found, the model will be reset.
            if(self.available_columns):
                self.available_column_model.loadData(self.available_columns)
                self.full_text_count.setText(str(self.full_data.shape[0]))
                self.display_selected_rows(None)
                self.update_statusbar.emit("CSV loaded.")
                self.select_all_btn.setEnabled(True)
                self.deselect_all_btn.setEnabled(True)
            else:
                exceptionWarning(f"No usable data found in {f_path}")
                self.logger.info(f"No usable data found in {f_path}")
                self.update_statusbar.emit("No usable data found in file")
            self.available_column_model.setCheckboxes(False)
            self.load_selected_data()
        except pd.errors.EmptyDataError as ede:
            exceptionWarning(
                exceptionTitle='Empty Data Error.\n', exception=ede)
        except Exception as e:
            self.logger.error("Error loading dataframe", exc_info=True)
            exceptionWarning(
                "Exception occured.  DataLoader.load_file.", exception=e)
            tb = traceback.format_exc()
            print(tb)

    def display_selected_rows(self, selection=None):
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
            offset = idx.row() * 2
            col_name = self.available_column_model.data(idx)
            label_col_name = col_name.split('__')[0] + '__actual'
            self.text_stats_groupbox.setTitle(col_name)
            question_data = self.full_data[col_name].fillna(
                value="unanswered")
            avg_num_words = get_avg_words_per_sample(str(question_data.values))
            self.current_question_count.setText(str(question_data.shape[0]))
            self.current_question_avg_word.setText("%.2f" % avg_num_words)

            self.graph.chartSingleClassFrequency(
                self.full_data[label_col_name].values.astype(int))
        except Exception as e:
            self.logger.error("Dataloader.display_selected_rows", exc_info=True)
            exceptionWarning(
                "Exception occured.  DataLoader.load_file.", exception=e)
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


    @pyqtSlot(pd.DataFrame)
    def update_data(self, data):
        self.load_data_btn.setEnabled(True)
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
                "Exception occured in DataLoader.applyPreprocessing", exc_info=True)
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
        apply_cols = [
            col for col in self.data.columns if col.endswith('_text')
        ]
        self.data[apply_cols] = self.data[apply_cols].applymap(
            lambda x: processText(str(x), **self.options)
        )
        if self.options['spell_correction']:
            sentences = self.data[apply_cols].applymap(
                lambda x: str(x).split()
            ).values
            sc = SpellCheck(sentences, CONFIG.get('VARIABLES', 'TopKSpellCheck'))

            self.data[apply_cols] = self.data[apply_cols].applymap(
                lambda x: sc.correct_spelling(x)
            )
        # sys.stdout = sys.__stdout__
        self.preprocessing_complete.emit(self.data)

    def stop_thread(self):
        # TODO: Add funtionality to stop the thread
        self.quit()
        self.wait()
